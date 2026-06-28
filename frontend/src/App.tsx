import { useRef, useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatInput } from './components/ChatInput';
import { ChatMessage } from './components/ChatMessage';
import { TypingDots } from './components/TypingDots';
import { chatApiClient } from './lib/api';
import type { ChatMessage as ChatMessageType } from './types/chat';

const initialMessage: ChatMessageType = {
  id: 'welcome',
  role: 'assistant',
  content: 'Nhập câu hỏi để truy vấn YouMed Neo4j GraphRAG. Kết quả sẽ gồm answer, Cypher và evidence rows.',
};

function createId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessageType[]>([initialMessage]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50);
  };

  const resetChat = () => {
    setMessages([initialMessage]);
  };

  const handleSend = async (message: string) => {
    const userMessage: ChatMessageType = { id: createId(), role: 'user', content: message };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    scrollToBottom();

    try {
      const data = await chatApiClient.sendMessage(message, true);
      const assistantMessage: ChatMessageType = {
        id: createId(),
        role: 'assistant',
        content: data.answer || 'Không có câu trả lời.',
        cypher: data.cypher,
        rows: data.rows,
        rowCount: data.row_count,
        error: data.error,
        answerError: data.answer_error,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const assistantMessage: ChatMessageType = {
        id: createId(),
        role: 'assistant',
        content: 'Không gọi được backend FastAPI.',
        error: error instanceof Error ? error.message : String(error),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } finally {
      setLoading(false);
      scrollToBottom();
    }
  };

  return (
    <div className="app-shell">
      <Sidebar onNewChat={resetChat} />
      <main className="chat-shell">
        <header className="topbar">
          <div>
            <h1>YouMed GraphRAG Chat</h1>
            <p>React + FastAPI + Neo4j + LangChain</p>
          </div>
        </header>

        <section className="messages">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {loading && <TypingDots />}
          <div ref={bottomRef} />
        </section>

        <div className="input-area">
          <ChatInput disabled={loading} onSend={handleSend} />
          <div className="hint">Enter để gửi, Shift + Enter để xuống dòng.</div>
        </div>
      </main>
    </div>
  );
}
