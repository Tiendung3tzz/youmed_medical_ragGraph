import { FormEvent, KeyboardEvent, useState } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  disabled?: boolean;
  onSend: (message: string) => void;
}

export function ChatInput({ disabled, onSend }: ChatInputProps) {
  const [value, setValue] = useState('');

  const submit = () => {
    const message = value.trim();
    if (!message || disabled) return;
    setValue('');
    onSend(message);
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    submit();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <textarea
        value={value}
        disabled={disabled}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Hỏi dữ liệu YouMed GraphRAG..."
        rows={1}
      />
      <button type="submit" disabled={disabled || !value.trim()} aria-label="Send">
        <Send size={18} />
      </button>
    </form>
  );
}
