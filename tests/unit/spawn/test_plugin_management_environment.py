"""Plugin tools must inherit the same bound team identity as model tools."""
from fast_agent.spawn.config_reader import get_server_env


def test_plugin_management_receives_team_session_and_agent(monkeypatch):
    monkeypatch.setenv('TEAM_SESSION_ID','plugin-team')
    monkeypatch.setenv('JARVIS_RUNTIME_RPC_SOCKET','/tmp/plugin-rpc.sock')
    monkeypatch.setenv('SPAWN_REGISTRY_DB','/tmp/plugin-test.db')
    env=get_server_env('plugin_management',workspace_dir='/tmp/workspace',agent_name='Developer')
    assert env is not None
    assert env['TEAM_SESSION_ID']=='plugin-team'
    assert env['TEAM_MY_NAME']=='Developer'
    assert env['SPAWN_REGISTRY_DB']=='/tmp/plugin-test.db'
