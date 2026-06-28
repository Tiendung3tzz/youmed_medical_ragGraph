import { MessageSquare, Plus } from 'lucide-react';

interface SidebarProps {
  onNewChat: () => void;
}

export function Sidebar({ onNewChat }: SidebarProps) {
  return (
    <aside className="sidebar">
      <button className="new-chat" onClick={onNewChat}>
        <Plus size={16} />
        Cuộc trò chuyện mới
      </button>

      <div className="history-title">GraphRAG</div>
      <div className="history-item active">
        <MessageSquare size={16} />
        YouMed Neo4j Chat
      </div>

      <div className="sidebar-footer">
        <div className="pill">FastAPI</div>
        <div className="pill">Neo4j</div>
        <div className="pill">GraphCypherQAChain</div>
      </div>
    </aside>
  );
}
