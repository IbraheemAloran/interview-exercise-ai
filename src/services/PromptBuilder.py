import json
import logging
logger = logging.getLogger(__name__)

class PromptBuilder:
  # Available actions for ticket resolution
    ACTIONS = [
        "none",
        "escalate_to_abuse_team",
        "escalate_to_legal_team", 
        "escalate_to_sales_team",
        "follow_up_required"
    ]
    
    # System role defines assistant behavior and constraints
    SYSTEM_ROLE = """You are an expert support assistant with access to a knowledge base.

TASK: Answer customer support queries using ONLY the provided context documents.

CORE RULES:
1. Use ONLY information from the provided context documents
2. Do NOT make up, infer, or hallucinate information
3. If the answer is not in the context, explicitly state "This information is not available in our knowledge base" and Action is "follow_up_required"
4. Always cite the source document(s) you used
5. Choose an action from the provided list - do NOT infer actions
6. Return ONLY valid JSON - no markdown, no explanations, no additional text"""

    # JSON schema for valid responses
    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Clear answer based on context. Say unavailable if not found."
            },
            "references": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of document filenames cited"
            },
            "action_required": {
                "type": "string",
                "enum": ACTIONS,
                "description": "Recommended action from the provided list"
            }
        },
        "required": ["answer", "references", "action_required"]
    }
    
    # JSON schema for input context documents with relevance scores
    INPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The document text content"
            },
            "metadata": {
                "type": "object",
                "description": "Document metadata",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the source document"
                    }
                },
                "required": ["filename"]
            },
        },
        "required": ["text", "metadata"]
    }
    
    # Few-shot examples for in-context learning
    FEW_SHOT_EXAMPLES = [
        {
            "query": "My domain was suspended and I didn't get any notice. How can I reactivate it?",
            "context_summary": "Document: domain_suspension_policy.txt - Contains policy on domain suspensions and reactivation steps",
            "response": {
                "answer": "Your domain may have been suspended due to a violation of our policy or missing WHOIS information. To reactivate it, update your WHOIS details with current information and contact our support team.",
                "references": ["domain_suspension_policy.txt"],
                "action_required": "escalate_to_abuse_team"
            }
        },
        {
            "query": "I can't remember my password. What should I do?",
            "context_summary": "Document: account_recovery.txt - Provides account recovery and password reset procedures",
            "response": {
                "answer": "To recover your account, use the password reset link on the login page. If you don't receive the email, check your spam folder or contact support.",
                "references": ["account_recovery.txt"],
                "action_required": "follow_up_required"
            }
        }
    ]

    @classmethod
    def build_prompt(cls, query: str, context_docs: list[dict]) -> str:
        #builds prompt from all components
        try:
            # Validate inputs
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            
            if not context_docs or not isinstance(context_docs, list):
                raise ValueError("Context docs must be a non-empty list")
            
            logger.info(f"Building MCP prompt for query: {query[:50]}...")
            
            # Format context documents
            formatted_context = cls._format_context_documents(context_docs)
            
            # Format few-shot examples
            formatted_examples = cls._format_few_shot_examples()
            
            # Build complete prompt
            prompt = f"""{cls.SYSTEM_ROLE}

{'='*70}
OUTPUT SCHEMA (respond with valid JSON only):
{cls._format_json_schema()}

{'='*70}
AVAILABLE ACTIONS:
{cls._format_actions()}

{'='*70}
IN-CONTEXT EXAMPLES (follow this pattern):

{formatted_examples}

{'='*70}
KNOWLEDGE BASE DOCUMENTS:

{formatted_context}

{'='*70}
CUSTOMER QUERY:

{query}

{'='*70}
RESPONSE (JSON only):
"""
            
            logger.info("MCP prompt built successfully")
            return prompt
            
        except ValueError as e:
            logger.error(f"Invalid input for prompt building: {str(e)}")
            raise
        except Exception as e:
            logger.exception("Error building MCP prompt")
            raise
    
    @classmethod
    def _format_context_documents(cls, context_docs: list[dict]) -> str:
        #extract information from docs
        if not context_docs:
            return "[No documents provided]"
        
        formatted = ""
        for i, doc in enumerate(context_docs, 1):
            text = doc['metadata']['text']
            filename = doc['metadata']['metadata']['filename']
            
            
            # Cut long documents to save tokens
            if len(text) > 1000:
                text = text[:1000]
            
            formatted += f"\n[Document {i}: {filename}]\n"
            formatted += f"{text}\n"
            formatted += "-" * 50 + "\n"
        
        return formatted
    
    @classmethod
    def _format_few_shot_examples(cls) -> str:
        """Format few-shot examples for in-context learning"""
        formatted = ""
        
        for i, example in enumerate(cls.FEW_SHOT_EXAMPLES, 1):
            formatted += f"\nEXAMPLE {i}:\n"
            formatted += f"Context: {example['context_summary']}\n"
            formatted += f"Query: {example['query']}\n"
            formatted += f"Response: {cls._dict_to_json(example['response'])}\n"
            formatted += "-" * 50
        
        return formatted
    
    @classmethod
    def _format_actions(cls) -> str:
        """Format available actions list"""
        return "\n".join([f"  • {action}" for action in cls.ACTIONS])
    
    @classmethod
    def _format_json_schema(cls) -> str:
        """Format JSON schema for clarity"""
        return """{
  "answer": "string - your response based on context only",
  "references": ["array of document filenames used"],
  "action_required": "one of: none|escalate_to_abuse_team|escalate_to_legal_team|escalate_to_sales_team|follow_up_required"
}"""
    
    @classmethod
    def _dict_to_json(cls, data: dict) -> str:
        """Convert dict to JSON string"""
        return json.dumps(data, indent=2)
    