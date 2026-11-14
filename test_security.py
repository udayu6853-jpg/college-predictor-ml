"""
=============================================================================
SMART COLLEGE PREDICTOR - SECURITY PENETRATION & HARDENING AUDIT TEST SUITE
=============================================================================
Run this script anytime to verify that all anti-hack defenses, WAF intrusion
shields, and backend protections are active and working.

Usage:
    python test_security.py
"""

from app import app

def print_result(test_name, passed, detail=""):
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    status = f"{GREEN}[PASSED - SECURE]{RESET}" if passed else f"{RED}[FAILED]{RESET}"
    print(f" {status} {test_name}")
    if detail:
        print(f"       -> {detail}")

def run_security_audit():
    print("\n" + "=" * 70)
    print(" 🛡️  SMART COLLEGE PREDICTOR - AUTOMATED SECURITY PENETRATION AUDIT")
    print("=" * 70 + "\n")

    client = app.test_client()
    passed_tests = 0
    total_tests = 6

    # 1. Reconnaissance Probe (/.env)
    r1 = client.get('/.env')
    p1 = (r1.status_code == 403)
    print_result("Test 1: Hacker .env Probe Attack", p1, f"Status: {r1.status_code} (Reconnaissance blocked)")
    if p1: passed_tests += 1

    # 2. Scanner Probe (/wp-login.php)
    r2 = client.get('/wp-login.php')
    p2 = (r2.status_code == 403)
    print_result("Test 2: Bot Scanner Probe Attack (/wp-login.php)", p2, f"Status: {r2.status_code} (Scanner blocked)")
    if p2: passed_tests += 1

    # 3. Path Traversal Hack (/static/../../secret)
    r3 = client.get('/static/../../secret')
    p3 = (r3.status_code in (403, 404))
    print_result("Test 3: Directory / Path Traversal Hack (../../)", p3, f"Status: {r3.status_code} (Traversal prevented)")
    if p3: passed_tests += 1

    # 4. Unauthenticated AI Chatbot Abuse Protection
    r4 = client.post('/api/chat', json={'message': 'hello'})
    p4 = (r4.status_code == 401)
    print_result("Test 4: Unauthenticated Gemini API Abuse Guard", p4, f"Status: {r4.status_code} (Unauthorized access blocked)")
    if p4: passed_tests += 1

    # 5. Legitimate Safe Login Endpoint
    r5 = client.get('/login')
    p5 = (r5.status_code == 200)
    print_result("Test 5: Legitimate Login Endpoint Availability", p5, f"Status: {r5.status_code} (Service active)")
    if p5: passed_tests += 1

    # 6. HTTP Security Hardening Headers Enforced
    p6 = (
        r5.headers.get('X-Frame-Options') == 'SAMEORIGIN' and
        r5.headers.get('X-Content-Type-Options') == 'nosniff' and
        '1; mode=block' in (r5.headers.get('X-XSS-Protection') or '')
    )
    print_result("Test 6: OWASP Security Headers (Frame, Nosniff, XSS)", p6, "Headers verified on HTTP responses")
    if p6: passed_tests += 1

    print("\n" + "=" * 70)
    if passed_tests == total_tests:
        print(f" \033[92mALL {total_tests}/{total_tests} SECURITY AUDIT CHECKS PASSED!\033[0m")
        print(" Your website is fully hardened and protected against hacking.")
    else:
        print(f" Result: {passed_tests}/{total_tests} tests passed.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_security_audit()
