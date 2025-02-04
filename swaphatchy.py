import json
from web3 import Web3
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
AVAX_RPC_URL = os.getenv("AVAX_RPC_URL")
TRADERJOE_ROUTER = "0x18556DA13313f3532c54711497A8FedAC273220E"
HATCHY_TOKEN = "0x502580fc390606b47FC3b741d6D49909383c28a9"
WAVAX = "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7"
MY_ADDRESS = os.getenv("MY_ADDRESS")

# Connect to Avalanche RPC
web3 = Web3(Web3.HTTPProvider(AVAX_RPC_URL))

# Load TraderJoe Router ABI
with open("TraderJoeABI.json") as f:
    TRADERJOE_ABI = json.load(f)

# Load ERC-20 Token ABI (balanceOf is part of the ERC-20 standard)
with open("ERC20ABI.json") as f:  # Ensure this file exists with the standard ERC-20 ABI
    ERC20_ABI = json.load(f)

traderjoe = web3.eth.contract(address=TRADERJOE_ROUTER, abi=TRADERJOE_ABI)
hatchy_contract = web3.eth.contract(address=HATCHY_TOKEN, abi=ERC20_ABI)

# Swap Function
def swap_native_for_tokens(amount_in_avax, amount_out_min, token_out, num_swaps=5):
    try:
        deadline = int(time.time()) + 600
        nonce = web3.eth.get_transaction_count(MY_ADDRESS)
        txs = []
        for _ in range(num_swaps):
            path = {
                "pairBinSteps": [0],  # Confirm bin steps
                "versions": [0],      # Confirm version
                "tokenPath": [WAVAX, token_out]
            }
            txn = traderjoe.functions.swapExactNATIVEForTokens(
                amount_out_min,
                path,
                MY_ADDRESS,
                deadline
            ).build_transaction({
                'from': MY_ADDRESS,
                'value': web3.to_wei(amount_in_avax, 'ether'),
                'gas': 300000,
                'gasPrice': int(web3.eth.gas_price * 1.2),
                'nonce': nonce
            })
            signed_txn = web3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
            tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)  # Corrected here
            txs.append(tx_hash)
            print(f"Transaction sent: {web3.to_hex(tx_hash)}")
            nonce += 1
            time.sleep(2)
        return txs
    except Exception as e:
        print(f"Error during swap: {e}")
        return None

# Sell Hatchy Tokens for AVAX
def swap_tokens_for_native(amount_in_hatchy, amount_out_min_avax):
    try:
        deadline = int(time.time()) + 600
        path = {
            "pairBinSteps": [0],  # Confirm bin steps
            "versions": [0],      # Confirm version
            "tokenPath": [HATCHY_TOKEN, WAVAX]
        }
        nonce = web3.eth.get_transaction_count(MY_ADDRESS)
        txn = traderjoe.functions.swapTokensForExactNATIVE(
            amount_out_min_avax,
            amount_in_hatchy,
            path,
            MY_ADDRESS,
            deadline
        ).build_transaction({
            'from': MY_ADDRESS,
            'gas': 300000,
            'gasPrice': int(web3.eth.gas_price * 1.2),
            'nonce': nonce
        })
        signed_txn = web3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)  # Corrected here
        print(f"Transaction sent: {web3.to_hex(tx_hash)}")
        return tx_hash
    except Exception as e:
        print(f"Error during sell: {e}")
        return None

# Example Usage
if __name__ == "__main__":
    avax_per_buy = 0.27
    slippage_tolerance = 0.05
    expected_hatchy = 9000 * (10 ** 18)
    min_hatchy_out = int(expected_hatchy * (1 - slippage_tolerance))
    min_avax_out = web3.to_wei(0.15, 'ether')

    print("Executing buys...")
    swap_native_for_tokens(avax_per_buy, min_hatchy_out, HATCHY_TOKEN, num_swaps=5)

    print("Selling Hatchy tokens...")
    hatchy_balance = hatchy_contract.functions.balanceOf(MY_ADDRESS).call()  # Correct ABI used here
    swap_tokens_for_native(hatchy_balance, min_avax_out)
