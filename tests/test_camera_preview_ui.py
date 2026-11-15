from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"


def test_buffalo_s_camera_modal_shows_live_preview_guidance_and_statuses():
    html = DASHBOARD.read_text(encoding="utf-8")

    assert '<dialog id="camera-modal"' in html
    assert 'id="camera-preview"' in html
    assert 'class="camera-guide"' in html
    assert "Center your face inside the frame." in html
    for status in ("Camera connecting", "Camera ready", "Detecting face", "Face detected",
                   "Capturing", "Capture successful", "Capture failed"):
        assert status in html
    assert 'fetch("/api/camera/session/frame"' in html
    assert 'api("/api/camera/session",{method:"POST"})' in html
    assert 'api("/api/camera/session",{method:"DELETE"})' in html


def test_dashboard_uses_backend_preview_and_keeps_simulated_checkin_path():
    html = DASHBOARD.read_text(encoding="utf-8")

    assert "getUserMedia" not in html
    assert 'recognitionMode==="buffalo_s"' in html
    assert 'request={member_id:memberId,liveness_score:Number(byId("liveness").value)}' in html
    assert 'await api("/api/check-in",{method:"POST",body:JSON.stringify(request)})' in html


def test_member_actions_use_the_requested_beginner_friendly_labels():
    html = DASHBOARD.read_text(encoding="utf-8")

    for label in ("Pause Member", "Reactivate", "Disable Face ID", "Set Up Face ID",
                  "Scan Face", "Check In", "Sign Out", "Refresh List", "Register Member"):
        assert label in html
    for old_label in ("textContent=\"Suspend\"", "textContent=\"Activate\"",
                      "textContent=\"Revoke consent\"", "textContent=\"Capture face\"",
                      "Capture and evaluate", "Record consent and capture face", ">Sign out<", ">Refresh<"):
        assert old_label not in html


def test_revoked_consent_action_requires_explicit_confirmation_and_calls_restore_api():
    html = DASHBOARD.read_text(encoding="utf-8")

    assert 'm.consent_status==="revoked"' in html
    assert "Has ${name} clearly agreed to set up Face ID again? Confirm only after they have provided consent." in html
    assert 'api(`/api/members/${id}/restore-consent`' in html
    assert 'JSON.stringify({confirmed:true})' in html
