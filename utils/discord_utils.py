import os
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def send_discord_notification(
        message: str,
        webhook_url: Optional[str] = None,
        username: str = "Kaggle Bot",
        color: Optional[int] = None
) -> bool:
    """
    Send a notification to Discord via webhook.

    Args:
        message: The message to send
        webhook_url: Discord webhook URL (defaults to env variable)
        username: Bot username to display
        color: Embed color (e.g., 0x00FF00 for green, 0xFF0000 for red)

    Returns:
        bool: True if successful, False otherwise
    """
    webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')

    if not webhook_url:
        print("Warning: No Discord webhook URL configured")
        return False

    payload = {
        "username": username,
        "embeds": [{
            "description": message,
            "color": color
        }]
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")
        return False


def notify_script_complete(script_path: str, kernel_slug: str):
    """Send notification for completed script."""
    message = f"✅ **Script Completed**\n📄 `{script_path}`\n🔗 Kernel: `{kernel_slug}`"
    send_discord_notification(message, color=0x00FF00)


def notify_script_failed(script_path: str, kernel_slug: str, retry_count: int, max_retries: int):
    """Send notification for failed script."""
    if retry_count >= max_retries:
        message = f"❌ **Script Failed Permanently**\n📄 `{script_path}`\n🔗 Kernel: `{kernel_slug}`\n🔄 Retries exhausted: {retry_count}/{max_retries}"
        send_discord_notification(message, color=0xFF0000)
    else:
        message = f"⚠️ **Script Failed (Retrying)**\n📄 `{script_path}`\n🔗 Kernel: `{kernel_slug}`\n🔄 Retry: {retry_count}/{max_retries}"
        send_discord_notification(message, color=0xFFA500)


def notify_batch_summary(completed: int, failed: int, total: int, failed_scripts: list):
    """Send summary notification for batch execution."""
    failed_list = "\n".join([f"  • `{script}`" for script in failed_scripts]) if failed_scripts else "  None"
    message = (
        f"📊 **Kaggle Batch Execution Summary**\n"
        f"✅ Completed: {completed}/{total}\n"
        f"❌ Failed: {failed}/{total}\n"
        f"\n**Failed Scripts:**\n{failed_list}"
    )
    color = 0x00FF00 if failed == 0 else 0xFF0000
    send_discord_notification(message, color=color)
