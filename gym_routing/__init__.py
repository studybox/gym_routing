from gym.envs.registration import register
# Routing
# ----------------------------------------
register(
    id='Cspp-v0',
    entry_point='gym_routing.envs:CsppEnv'
)

