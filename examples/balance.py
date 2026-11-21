"""Prints the balance of the account that owns the API key.

Run it with your API key in DEVUP_API_KEY:

    python examples/balance.py
"""

from devupai import DevupAI


def main() -> None:
    with DevupAI() as client:
        balance = client.balance.retrieve()
    print(f"Balance: {balance.balance_dzd} DZD")


if __name__ == "__main__":
    main()
