import type { ChatResponse } from '../types/chat';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export class ChatApiClient {
  constructor(private readonly baseUrl: string = API_BASE_URL) {}

  async sendCypherMessage(message: string, includeDebug = true): Promise<ChatResponse> {
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, include_debug: includeDebug }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API ${res.status}: ${text}`);
    }

    return res.json();
  }
  
  async sendMessage(message: string, includeDebug = true): Promise<ChatResponse> {
    // Hybrid endpoint: Qdrant returns section_id, Neo4j enriches by those IDs.
    const res = await fetch(`${this.baseUrl}/api/chat/hybrid`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, include_debug: includeDebug }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API ${res.status}: ${text}`);
    }

    return res.json();
  }
}

export const chatApiClient = new ChatApiClient();
