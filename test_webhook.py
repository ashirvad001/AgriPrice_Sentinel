import urllib.request, urllib.parse, urllib.error
import json

queries = ["wheat Amritsar", "subscribe wheat Amritsar", "history wheat Amritsar"]
for q in queries:
    print(f"\n--- Testing: {q} ---")
    data = urllib.parse.urlencode({'From': 'whatsapp:+919876543210', 'Body': q}).encode('utf-8')
    req = urllib.request.Request('http://localhost:8000/api/v1/whatsapp/webhook', data=data, method='POST')
    try:
        response = urllib.request.urlopen(req)
        print("SUCCESS:\n")
        print(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"FAILED (HTTP {e.code}):\n")
        print(e.read().decode('utf-8'))
    except Exception as e:
        print(e)
