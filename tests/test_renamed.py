"""Before 1.2 the tool was Profile Match: its data folder, settings and command keep working."""

import sys

from resume_shortlister import app
from resume_shortlister.config import Settings, default_home, tool_env


def test_an_existing_profile_match_folder_is_used_until_a_new_one_exists(tmp_path):
    assert default_home(tmp_path) == tmp_path / ".resume-shortlister"  # a new user
    (tmp_path / ".profile-match").mkdir()
    assert default_home(tmp_path) == tmp_path / ".profile-match"  # someone upgrading keeps their data
    (tmp_path / ".resume-shortlister").mkdir()
    assert default_home(tmp_path) == tmp_path / ".resume-shortlister"


def test_old_setting_names_still_work_and_new_ones_win(clean_env, tmp_path):
    for name in ("RESUME_SHORTLISTER_HOME", "PROFILE_MATCH_HOME"):
        clean_env.delenv(name, raising=False)
    clean_env.setenv("PROFILE_MATCH_HOME", str(tmp_path / "old"))
    assert tool_env("HOME") == str(tmp_path / "old")
    assert Settings.from_env().sessions_db == str(tmp_path / "old" / "sessions.sqlite")
    clean_env.setenv("RESUME_SHORTLISTER_HOME", str(tmp_path / "new"))
    assert Settings.from_env().sessions_db == str(tmp_path / "new" / "sessions.sqlite")
    clean_env.setenv("PROFILE_MATCH_TOKEN", "dev")
    assert tool_env("TOKEN") == "dev"


def test_the_old_command_still_works_and_says_so(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["/usr/local/bin/profile-match", "--version"])
    assert app.main() == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("resume-shortlister ")
    assert "profile-match is now resume-shortlister" in captured.err

    monkeypatch.setattr(sys, "argv", ["/usr/local/bin/resume-shortlister", "--version"])
    assert app.main() == 0
    assert capsys.readouterr().err == ""
