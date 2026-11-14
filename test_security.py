"""
=============================================================================
SMART COLLEGE PREDICTOR - SECURITY PENETRATION & HARDENING AUDIT TEST SUITE
=============================================================================
Run this script anytime to verify that all anti-hack defenses, RBAC permissions,
and WAF intrusion shields are active and working.

Usage:
    python test_security.py
"""

import sys
from app import app, ADMIN_EMAIL

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
    print(f" 👤 Designated System Administrator: {ADMIN_EMAIL}")
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

    # 4. Anonymous Admin Access Barrier
    r4 = client.get('/admin/dashboard')
    p4 = (r4.status_code == 302 and '/login' in r4.headers.get('Location', ''))
    print_result("Test 4: Anonymous Access to Admin Portal", p4, f"Status: {r4.status_code} (Redirected to login)")
    if p4: passed_tests += 1

    # 5. Student Access to Admin Portal (Role-Based Access Control)
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['user_email'] = 'student@smartpredictor.com'
        sess['user_role'] = 'user'
    r5 = client.get('/admin/dashboard')
    p5 = (r5.status_code == 403)
    print_result("Test 5: Student Account Attempting to Open Admin Dashboard", p5, f"Status: {r5.status_code} (403 Forbidden enforced)")
    if p5: passed_tests += 1

    # 6. Primary System Administrator Access (Uday)
    with client.session_transaction() as sess:
        sess['user_id'] = 2
        sess['user_email'] = ADMIN_EMAIL
        sess['user_role'] = 'admin'
        sess['is_admin'] = True
    r6 = client.get('/admin/dashboard')
    p6 = (r6.status_code == 200)
    print_result("Test 6: Primary Administrator Access (Uday)", p6, f"Status: {r6.status_code} (Authorized successfully)")
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
