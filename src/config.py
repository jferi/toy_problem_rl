class EnvConfig:
    BATCH_SIZE = 128
    S_MEAN = 100.0
    VOL = 2.0
    KAPPA = 0.1
    HALF_BA = 1.0
    RISK_AV = 0.02
    LOOKBACK = 10
    MAX_STEPS = 200


class AgentConfig:
    LEARNING_RATE = 0.0001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE = 10
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 100000


class TrainingConfig:
    NUM_EPISODES = 300
    BATCH_SIZE = 128
    MAX_STEPS = 200
    LOOKBACK = 10
    PRINT_EVERY = 20


class NetworkConfig:
    HIDDEN_SIZE = 128
    NUM_ACTIONS = 3
    LOOKBACK = 10

class EvaluationConfig:
    NUM_EPISODES = 10
    BATCH_SIZE = 1
    VISUALIZE = True
    MAX_STEPS = 200