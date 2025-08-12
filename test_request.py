import requests

# IMPORTANT: Replace this with your new ngrok URL
url = 'https://3b0bca65ed5e.ngrok-free.app/api/'
files = {
    'questions.txt': open('questions.txt', 'rb')
}

print(f"Attempting to send request to {url}")

try:
    response = requests.post(url, files=files)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("\nRequest was successful!")
        print("Response from server:")
        print(response.json())
    else:
        print(f"\nRequest failed with status code: {response.status_code}")
        print("Response from server:")
        print(response.json())

except requests.exceptions.RequestException as e:
    print(f"\nAn error occurred during the request: {e}")
