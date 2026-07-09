import type { ChatMessage as ChatMessageType } from '../types/chat';

interface ChatMessageProps {
  message: ChatMessageType;
}

function EvidenceRows({ rows }: { rows?: Record<string, unknown>[] }) {
  if (!rows || rows.length === 0) return null;
  return (
    <details className="debug-block">
      <summary>Evidence rows ({rows.length})</summary>
      <pre>{JSON.stringify(rows, null, 2)}</pre>
    </details>
  );
}

function QdrantHits({ hits }: { hits?: Record<string, unknown>[] }) {
  if (!hits || hits.length === 0) return null;
  return (
    <details className="debug-block">
      <summary>Qdrant hits ({hits.length})</summary>
      <pre>{JSON.stringify(hits, null, 2)}</pre>
    </details>
  );
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div className="avatar">{isUser ? 'U' : 'AI'}</div>
      <div className="message-body">
        <div className="message-content">{message.content}</div>

        {message.error && <div className="error-box">Graph error: {message.error}</div>}
        {message.answerError && <div className="error-box">Answer error: {message.answerError}</div>}

        {!isUser && message.retrievalMode && (
          <div className="meta-line">Retrieval mode: {message.retrievalMode}</div>
        )}

        {!isUser && message.cypher && (
          <details className="debug-block">
            <summary>Cypher</summary>
            <pre>{message.cypher}</pre>
          </details>
        )}
        
        {!isUser && <QdrantHits hits={message.qdrantHits as Record<string, unknown>[] | undefined} />}
        {!isUser && <EvidenceRows rows={message.rows as Record<string, unknown>[] | undefined} />}
      </div>
    </div>
  );
}
