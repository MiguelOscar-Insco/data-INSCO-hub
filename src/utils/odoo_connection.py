import requests
import json

def connect_to_odoo_api(path, parameters):
    """
    Connects to an Odoo instance using the RESTful API.
    Args:
        path (str): The API endpoint path.
        parameters (dict): The request parameters.
    Returns:
        dict or None: The JSON response from the API, or None if an error occurs.
    """
    # Function code goes here

    url = 'http://example.com:8069/'
    db = 'sample_db'
    username = 'sample_user'
    password = 'sample_password'

    session_url = f'{url}/web/session/authenticate'
    data = {
        'jsonrpc': '2.0',
        'method': 'call',
        'params': {
            'db': db,
            'login': username,
            'password': password,
        }
    }
    session_response = requests.post(session_url, json=data)
    session_data = session_response.json()

    if session_data.get('result') and session_response.cookies.get('session_id'):
        session_id = session_response.cookies['session_id']
    else:
        print(f'Error: Failed to authenticate - {session_data.get("error")}')
        return None


headers = {
    'Content-Type': 'application/json',
    'Cookie': f"session_id={session_id}",
}
