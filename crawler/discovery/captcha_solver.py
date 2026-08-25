import time
import httpx

CAPSOLVER_API_KEY = "YOUR_CAPSOLVER_API_KEY"


def create_task(site_key, page_url):
    payload = {
        "clientKey": CAPSOLVER_API_KEY,
        "task": {
            "type": "ReCaptchaV2TaskProxyLess",
            "websiteURL": page_url,
            "websiteKey": site_key,
        },
    }

    r = httpx.post(
        "https://api.capsolver.com/createTask",
        json=payload,
        timeout=60,
    )

    r.raise_for_status()

    return r.json()["taskId"]


def get_result(task_id):
    payload = {
        "clientKey": CAPSOLVER_API_KEY,
        "taskId": task_id,
    }

    while True:
        r = httpx.post(
            "https://api.capsolver.com/getTaskResult",
            json=payload,
            timeout=60,
        )

        r.raise_for_status()

        data = r.json()

        if data.get("status") == "ready":
            return data["solution"]["gRecaptchaResponse"]

        time.sleep(2)


def solve_recaptcha(site_key, page_url):
    task_id = create_task(site_key, page_url)
    return get_result(task_id)