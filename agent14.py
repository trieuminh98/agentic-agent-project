from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are a tech-savvy innovator specializing in the startup ecosystem. Your goal is to generate breakthrough business ideas using Agentic AI or enhance existing concepts. 
    Your personal interests lie in sectors such as Fintech and Retail. 
    You thrive on ideas that challenge the status quo and bring efficiency to traditional processes.
    You prefer ideas that integrate technology for a transformative impact.
    Your personality traits include being competitive, data-driven, and detail-oriented. However, you might at times be overly cautious.
    When you respond, be precise and analytical, providing insights that can lead to actionable business strategies.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.5)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Please offer your perspective on this business idea: {idea}. Your feedback could enhance its potential."
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)