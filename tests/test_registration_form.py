from html.parser import HTMLParser
from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"


class FormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.forms = {}
        self.current_form = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id"):
            self.current_form = attrs
            self.forms[attrs["id"]] = attrs
        elif tag == "input" and attrs.get("id"):
            self.inputs[attrs["id"]] = attrs

    def handle_endtag(self, tag):
        if tag == "form":
            self.current_form = None


def test_contact_fields_are_editable_without_autofill_values():
    html = DASHBOARD.read_text(encoding="utf-8")
    parser = FormParser()
    parser.feed(html)

    phone = parser.inputs["phone"]
    email = parser.inputs["email"]
    assert "readonly" not in phone and "disabled" not in phone
    assert "readonly" not in email and "disabled" not in email
    assert "value" not in phone and "value" not in email
    assert "placeholder" not in phone
    assert phone["name"] == "phone" and phone["autocomplete"] == "off"
    assert email["name"] == "email" and email["autocomplete"] == "off"
    assert phone["pattern"] == r"\+[1-9][0-9]{7,14}" and "required" in phone
    assert email["type"] == "email" and "required" not in email
    assert parser.forms["enroll-form"]["autocomplete"] == "off"
    assert ".field input:-webkit-autofill" in html


def test_enrollment_submission_reads_latest_contact_input_values():
    html = DASHBOARD.read_text(encoding="utf-8")
    assert 'byId("phone").value' in html
    assert 'byId("email").value||null' in html
    assert 'addEventListener("input"' in html
    assert 'byId("phone").value=' not in html
    assert 'byId("email").value=' not in html
    assert 'const request={full_name:byId("full-name").value,phone:byId("phone").value||null,email:byId("email").value||null' in html
    refresh_source = html.split("async function refresh(){", 1)[1].split("\n  }", 1)[0]
    assert 'byId("phone").value' not in refresh_source
    assert 'byId("email").value' not in refresh_source


def test_real_demo_dashboard_displays_identity_membership_and_access_tone():
    html = DASHBOARD.read_text(encoding="utf-8")

    assert "Member ID ${result.member_id||\"—\"}" in html
    assert "membership ${result.membership_status||\"—\"}" in html
    assert "Unknown Person" in html
    assert "access-granted" in html
    assert "access-denied" in html
    assert "access-unknown" in html
    assert "checkin-consent" in html
    assert "consent_confirmed:true" in html
