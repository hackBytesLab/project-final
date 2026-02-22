import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


def send_line_alert(channel_access_token, message, to_user_id=None, timeout=10):
    if not channel_access_token:
        raise ValueError("LINE channel access token is required")

    if to_user_id:
        endpoint = "https://api.line.me/v2/bot/message/push"
        payload = {
            "to": to_user_id,
            "messages": [{"type": "text", "text": message}],
        }
        mode = "push"
    else:
        endpoint = "https://api.line.me/v2/bot/message/broadcast"
        payload = {"messages": [{"type": "text", "text": message}]}
        mode = "broadcast"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {channel_access_token}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return mode, resp.status, body
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LINE API error {e.code}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"LINE API connection error: {e}") from e


def main():
    parser = argparse.ArgumentParser(
        description="Send a test LINE message via broadcast or push API."
    )
    parser.add_argument(
        "--token",
        default=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"),
        help="LINE channel access token. Defaults to env LINE_CHANNEL_ACCESS_TOKEN.",
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("LINE_USER_ID"),
        help="Target user/group/room id. Required only for --mode push.",
    )
    parser.add_argument(
        "--mode",
        choices=["broadcast", "push", "auto"],
        default="broadcast",
        help="broadcast=send to all followers, push=send to --user-id, auto=push when user id exists.",
    )
    parser.add_argument(
        "--message",
        default=f"[LINE TEST] {time.strftime('%Y-%m-%d %H:%M:%S')}",
        help="Message text to send.",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved mode and payload target without calling LINE API.",
    )
    args = parser.parse_args()

    if not args.token:
        print("ERROR: Missing token. Set --token or LINE_CHANNEL_ACCESS_TOKEN.", file=sys.stderr)
        raise SystemExit(2)

    if args.mode == "push":
        if not args.user_id:
            print("ERROR: --mode push requires --user-id (or LINE_USER_ID).", file=sys.stderr)
            raise SystemExit(2)
        to_user_id = args.user_id
    elif args.mode == "auto":
        to_user_id = args.user_id if args.user_id else None
    else:
        to_user_id = None

    resolved_mode = "push" if to_user_id else "broadcast"
    if args.dry_run:
        target = to_user_id if to_user_id else "<broadcast>"
        print(f"[DRY-RUN] mode={resolved_mode}, target={target}, message={args.message}")
        return

    try:
        mode, status, body = send_line_alert(
            channel_access_token=args.token,
            message=args.message,
            to_user_id=to_user_id,
            timeout=args.timeout,
        )
    except Exception as e:
        print(f"[LINE TEST] failed: {e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[LINE TEST] mode={mode}, status={status}")
    if body:
        print(body)


if __name__ == "__main__":
    main()
