# File: main.py
from __future__ import annotations

import os
import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.smtp.operators.smtp import EmailOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.task.trigger_rule import TriggerRule

from src.model_development import (
    load_data,
    data_preprocessing,
    separate_data_outputs,
    build_model,
    load_model,
)


# ---------- Notification callbacks ----------
def notify_success(context):
    """Record task success; EmailOperator handles completion emails."""
    task_id = context["task_instance"].task_id
    print(f"Task succeeded: {task_id}")


def notify_failure(context):
    """Send a failure alert only to the configured SMTP account."""
    import smtplib
    from email.message import EmailMessage

    sender = os.environ.get("AIRFLOW__SMTP__SMTP_USER")
    recipient = os.environ.get("AIRFLOW__SMTP__SMTP_MAIL_FROM")
    password = os.environ.get("AIRFLOW__SMTP__SMTP_PASSWORD")

    if not ENABLE_EMAIL or not all([sender, recipient, password]):
        print("Failure email skipped: notifications not configured")
        return

    if sender.strip().lower() != recipient.strip().lower():
        raise ValueError("Failure notification blocked: recipient mismatch")

    task_instance = context["task_instance"]

    message = EmailMessage()
    message["Subject"] = "Airflow Lab 2 - Task Failed"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(
        f"DAG: {task_instance.dag_id}\n"
        f"Task: {task_instance.task_id}\n"
        f"Run: {context['run_id']}\n"
        f"State: failed\n"
    )

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(sender, password)
        smtp.send_message(
            message,
            from_addr=sender,
            to_addrs=[recipient],
        )

    print("Failure notification sent to configured account")


# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
    "on_failure_callback": notify_failure,
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Airflow_Lab2",
    default_args=default_args,
    description="Airflow-Lab2 DAG Description",
    schedule="@daily",
    catchup=False,
    tags=["example"],
    owner_links={"Ramin Mohammadi": "https://github.com/raminmohammadi/MLOps/"},
    max_active_runs=1,
)

# ---------- Tasks ----------
owner_task = BashOperator(
    task_id="task_using_linked_owner",
    bash_command="echo 1",
    owner="Ramin Mohammadi",
    dag=dag,
)

# Email notifications use only the locally configured address.
# Disabled by default until SMTP delivery is explicitly tested.
EMAIL_RECIPIENT = os.getenv("AIRFLOW__SMTP__SMTP_USER")
ENABLE_EMAIL = os.getenv("ENABLE_LAB2_EMAIL", "false").lower() == "true"

if ENABLE_EMAIL:
    if not EMAIL_RECIPIENT:
        raise ValueError("Email enabled but SMTP recipient is missing")

    send_email = EmailOperator(
        task_id="send_email",
        to=EMAIL_RECIPIENT,
        subject="Airflow Lab 2 - Pipeline Completed",
        html_content="<p>Airflow Lab 2 model training completed successfully.</p>",
        dag=dag,
    )


load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="data_preprocessing_task",
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

separate_data_outputs_task = PythonOperator(
    task_id="separate_data_outputs_task",
    python_callable=separate_data_outputs,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id="build_save_model_task",
    python_callable=build_model,
    op_args=[separate_data_outputs_task.output, "model.sav"],
    dag=dag,
)

load_model_task = PythonOperator(
    task_id="load_model_task",
    python_callable=load_model,
    op_args=[separate_data_outputs_task.output, "model.sav"],
    dag=dag,
)

# Fire-and-forget trigger so this DAG can finish cleanly.
trigger_dag_task = TriggerDagRunOperator(
    task_id="my_trigger_task",
    trigger_dag_id="Airflow_Lab2_Flask",
    conf={"message": "Data from upstream DAG"},
    reset_dag_run=False,
    wait_for_completion=False,          # don't block
    trigger_rule=TriggerRule.ALL_SUCCESS,  # still run even if something upstream fails
    dag=dag,
)

# ---------- Dependencies ----------
owner_task >> load_data_task >> data_preprocessing_task >> \
    separate_data_outputs_task >> build_save_model_task >> \
    load_model_task >> trigger_dag_task

if ENABLE_EMAIL:
    load_model_task >> send_email
