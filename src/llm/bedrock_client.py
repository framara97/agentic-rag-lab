import json
import boto3
from typing import Optional


class BedrockLLM:
    """Client per interagire con Amazon Bedrock usando Claude."""
    
    def __init__(
        self,
        profile_name: str = "personal",
        region_name: str = "eu-west-1",
        model_id: str = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0"
    ):
        """
        Inizializza il client Bedrock.
        
        Args:
            profile_name: Nome del profilo AWS
            region_name: Regione AWS
            model_id: ID del modello Bedrock
        """
        session = boto3.Session(profile_name=profile_name)
        self.client = session.client("bedrock-runtime", region_name=region_name)
        self.model_id = model_id
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        Genera una risposta dal modello Claude.
        
        Args:
            prompt: Il prompt da inviare al modello
            max_tokens: Numero massimo di token nella risposta
            
        Returns:
            La risposta generata dal modello
        """
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
