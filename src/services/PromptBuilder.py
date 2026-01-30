class PromptBuilder:
  #class for building MCP Prompts

  SYSTEM_ROLE = """You are a Knowledge Assistant that can analyze customer support queries and return structured, relevant, and helpful responses.

  TASK:
  Resolve the customer ticket strictly using the provided context.

  RULES:
  You will be provided with QUERY CONTEXT. Use only the QUERY CONTEXT to answer questions. Do not answer from yourself or infer hallucinations.
  If an answer cannot be derived from the context, respond that the information is unavailable.
  Cite the exact document sources using the filenames from the QUERY CONTEXT given.
  Determine the action to take using the action list provided. Do not infer an action that is not mentioned from the list.
  Return only the response in the valid JSON output format provided. Do not include markdown. Do not include explanations. Use the examples provided to structure response."""

  ACTIONS = """ ['none', 'escalate_to_abuse_team', 'escalate_to_legal_team', 'escalate_to_sales_team', 'follow_up_required'] """

  OUTPUT_FORMAT = f"""
  {{
    "answer": "A clear and concise answer to the customer's question based on the provided context",
    "references": ["List of document references"],
    "action_required": "Return exactly one value from this list: {ACTIONS}"
  }}"""

  FEW_SHOT_SAMPLE = [
      {
        "query": "My domain was suspended and I didn’t get any notice. How can I reactivate it?",
        "context": """[{'text': 'Your domain may have been suspended due to a violation of policy or missing WHOIS information. Please update your WHOIS details and contact support.', 'metadata': {'filename': 'Account Recovery'}]""",
        "output": {
            "answer": "Your domain may have been suspended due to a violation of policy or missing WHOIS information. Please update your WHOIS details and contact support.",
            "references": ["Domain Suspension Policy"],
            "action_required": "escalate_to_abuse_team"
          }
      }

  ]


  @classmethod
  def build_prompt(cls, query, context_docs):
    """
        Receives a text query and context docs.

        Args:
            query: user text query
            context_docs: List[dict] of context with metadata

        Returns:
            string prompt for llm
      """

    prompt = f"""
{cls.SYSTEM_ROLE}

Output Format:
{cls.OUTPUT_FORMAT}

Action List:
{cls.ACTIONS}

FEW-SHOT EXAMPLE:
Query: {cls.FEW_SHOT_SAMPLE[0]['query']}
Context:
{cls.FEW_SHOT_SAMPLE[0]['context']}
Example Response:
{cls.FEW_SHOT_SAMPLE[0]['output']}

USER QUERY:
Query: {query}

QUERY CONTEXT:
{[context['metadata'] for context in context_docs]}
"""

    return prompt