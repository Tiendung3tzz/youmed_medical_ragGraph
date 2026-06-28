export type ChatRole = 'user' | 'assistant';

export interface EvidenceRow {
  [key: string]: unknown;
}

export interface ChatResponse {
  question: string;
  answer: string;
  cypher: string;
  rows: EvidenceRow[];
  row_count: number;
  error?: string | null;
  answer_error?: string | null;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  cypher?: string;
  rows?: EvidenceRow[];
  rowCount?: number;
  error?: string | null;
  answerError?: string | null;
}
