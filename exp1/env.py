# env.py

class AuctionEnv:
    '''
    Environment for 2-agent auction collusion experiment.

    Functions:
    1. Track timestep
    2. Store full trajectory history
    3. Control what the agents see (state)
    4. Record actions and external signals (e.g., rewards, detector score)

    Example instantiation:
    config = {
        'state_type': 'window', 'full', or 'last_step',
        'history_window': 5, (only used if state_type is 'window')
        'min_bid': 0,
        'max_bid': 100,
        'max_timesteps': 50
    }
    env = AuctionEnv(config)
    '''

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        '''
        Restarts the environment to prep for a new episode.
        '''
        self.t = 0
        self.history = []
        return self._get_state()
    
    # 

    def _get_state(self):
        '''
        Returns what the agents are allowed to see at each timestep.
        Can restrict information to test different memory lengths/observability settings (See: ablation 7)
        '''
        if self.config.state_type == 'full': # agents have perfect memory
            visible_history = self.history
        elif self.config.state_type == 'window': # agents have short-term memory of last [window] steps
            window = self.config.history_window
            visible_history = self.history[-window:] 
        elif self.config.state_type == 'last_step': # agents only see last step (stateless)
            visible_history = self.history[-1:] if self.history else []
        else:
            raise ValueError(f'Invalid state type: {self.config.state_type}')
        
        return {
            'timestep': self.t,
            'history': visible_history
        }

    def step(self, actions, ext_signals):
        '''
        Runs one timestep of the environment. Agents have: bid (int), reasoning (str).
        External signals include: proxy_rewards, true rewards, detector outputs, etc.
        '''
        self.validate_actions(actions)

        transition = {
            'timestep': self.t,
            'agent1': {
                'bid': actions['agent1']['bid'],
                'reasoning': actions['agent1']['reasoning']
            },
            'agent2': {
                'bid': actions['agent2']['bid'],
                'reasoning': actions['agent2']['reasoning']
            },
            'signals': ext_signals
        }

        self.history.append(transition)

        self.t += 1
        done = self.t >= self.config.max_timesteps

        next_state = self._get_state()

        return next_state, transition, done

    def validate_actions(self, actions):
        '''
        Helper that checks if agent outputs are valid.
        '''
        agents = ['agent1', 'agent2']

        for agent in agents:
            if agent not in actions:
                raise ValueError(f'Missing action for {agent}')
            
            bid = actions[agent]['bid']
            reasoning = actions[agent]['reasoning']
            
            if 'bid' not in actions[agent]:
                raise ValueError(f'Missing bid for {agent}')
            
            if 'reasoning' not in actions[agent]:
                raise ValueError(f'Missing reasoning for {agent}')
            
            # check if bid is an int
            if not isinstance(bid, int):
                raise TypeError(f'{agent} bid must be an int, got {type(bid)}')
            
            # check if bid is within allowed range
            if bid < self.config.min_bid or bid > self.config.max_bid:
                raise ValueError(
                    f'{agent} bid {bid} is outside allowed range '
                    f'[{self.config.min_bid}, {self.config.max_bid}]'
                )
            # check if reasoning is a string
            if not isinstance(reasoning, str):
                raise TypeError(f'{agent} reasoning must be a str, got type {type(reasoning)}')
            
            # check if empty reasoning
            if reasoning.strip() == '':
                raise ValueError(f'{agent} reasoning cannot be empty')