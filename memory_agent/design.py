class MemoryAssistant:
    def __init__(self, user_id: str = "demo_user"):
        self.user_id = user_id
        self.agent = Agent(
            system_prompt=MEMORY_SYSTEM_PROMPT,
            tools=[mem0_memory, use_llm],
        )

    def store_memory(self, content: str) -> Dict[str, Any]:
        # Implementation...

    def retrieve_memories(self, query: str, min_score: float = 0.3, max_results: int = 5) -> List[Dict[str, Any]]:
        # Implementation...

    def list_all_memories(self) -> List[Dict[str, Any]]:
        # Implementation...

    def generate_answer_from_memories(self, query: str, memories: List[Dict[str, Any]]) -> str:
        # Implementation...

    def process_input(self, user_input: str) -> str:
        # Implementation...