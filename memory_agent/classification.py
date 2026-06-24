def process_input(self, user_input: str) -> str:
    # Check if this is a memory storage request
    if user_input.lower().startswith(("remember ", "note that ", "i want you to know ")):
        content = user_input.split(" ", 1)[1]
        self.store_memory(content)
        return f"I've stored that information in my memory."

    # Check if this is a request to list all memories
    if "show" in user_input.lower() and "memories" in user_input.lower():
        all_memories = self.list_all_memories()
        # ... process and return memories list ...

    # Otherwise, retrieve relevant memories and generate a response
    relevant_memories = self.retrieve_memories(user_input)
    return self.generate_answer_from_memories(user_input, relevant_memories)