
# MA-survival-n-communication

Please refer this [link](https://github.com/KRLGroup/gym-ma-survival-2d) for the environment.

## Adding Communication
### One discrete and One continuous signal
- A discrete(D) and continuous(C) signal.
- The idea is that when D=1, the agent sends signal C to its teammate, and if D=0, then signal C is masked. Thus, each agent would have a signal that it could send (part of its action), and a signal it received from its teammate (part of its observation).

#### Original State and Action Space
For 1 Agent:
- `action_space = spaces.MultiDiscrete([3,3,3,2,2,2])`
- obs_space(Agent's observation only):
    - `agent_size = 1+1+3+3`
    - `agent_size +=1 if has teams`
    - Thus, `agent_size = 1+1(team_id)+1+3+3`
- Remaining observations, such as zone, objects, heals etc remains as it is.


#### New State and Action Space
For 1 Agent:
- `discrete_action_space = spaces.MultiDiscrete([3,3,3,2,2,2,2])`
- `cont_action_space = spaces.Box(-1, 1, (1,))`
- `actions_spaces = (single_action_space,cont_action_space,)`
- obs_space(Agent's observation only):
    - `agent_size = 1+1+3+3`
    - `agent_size +=1` if has teams
    - `agent_size += 1` if has communication
    - Thus, `agent_size = 1+1(team_id)+1+3+3+1(comms)`
- Remaining observations, such as zone, objects, heals etc remain same as original environment.