import prompt
import util
import config


class Service:
    def __init__(self):
        super(Service, self).__init__()
        self.util = util.Util()
        self.configs = config.ConfigParser()
        self.agent = self.util.initialize_agent()

    def reasoning_action_answer(self, message, history):
        # 用户问题 -> Agent -> Tools -> Chain
        return self.agent.run(message)
