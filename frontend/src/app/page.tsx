"use client";

import { useState, useRef, useEffect } from "react";
import { 
  Image as ImageIcon, Globe, Send, Bot, User, FileText, ExternalLink, Database, BookOpen
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import dynamic from "next/dynamic";

const DataSphere = dynamic(() => import("./components/DataSphere"), { ssr: false });

interface Source {
  text: string;
  metadata: {
    title?: string;
    url?: string;
    citation?: string;
    arxiv_id?: string;
  };
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

export default function ChatApp() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [greeting, setGreeting] = useState("Hello, Researcher.");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) setGreeting("Good Morning, Researcher.");
    else if (hour >= 12 && hour < 17) setGreeting("Good Afternoon, Researcher.");
    else if (hour >= 17 && hour < 21) setGreeting("Good Evening, Researcher.");
    else setGreeting("Good Night, Researcher.");
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (text: string = input) => {
    if (!text.trim() || isLoading) return;

    const userMessage: Message = { id: Date.now().toString(), role: "user", content: text };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    const assistantId = (Date.now() + 1).toString();
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", sources: [] },
    ]);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, stream: true }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      
      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n\n");
          
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const dataStr = line.substring(6);
              if (dataStr === "[DONE]") {
                setIsLoading(false);
                break;
              }
              
              try {
                const data = JSON.parse(dataStr);
                if (data.type === "sources") {
                  setMessages((prev) => 
                    prev.map((msg) => 
                      msg.id === assistantId ? { ...msg, sources: data.data } : msg
                    )
                  );
                } else if (data.type === "token") {
                  setMessages((prev) => 
                    prev.map((msg) => 
                      msg.id === assistantId ? { ...msg, content: msg.content + data.data } : msg
                    )
                  );
                } else if (data.type === "error") {
                  console.error("Error from server:", data.data);
                  setIsLoading(false);
                }
              } catch (e) {
                console.error("Error parsing JSON:", e, dataStr);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), role: "assistant", content: "Sorry, there was an error processing your request. Please ensure the backend is running." }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="brand-title">ML Research Assistant</div>
        
        <div className="stats-row">
          <div className="stat-box">
            <Database size={20} />
            <div className="stat-value">40+</div>
            <div className="stat-label">Papers</div>
          </div>
          <div className="stat-box">
            <BookOpen size={20} />
            <div className="stat-value">HyDE</div>
            <div className="stat-label">Retrieval</div>
          </div>
        </div>

        <div className="quick-ask-section">
          <div className="quick-ask-title">Quick Ask</div>
          <button className="quick-ask-btn" onClick={() => handleSend("What is the key innovation in the Transformer?")}>
            What is the key innovation in the Transformer?
          </button>
          <button className="quick-ask-btn" onClick={() => handleSend("How does LoRA reduce trainable parameters?")}>
            How does LoRA reduce trainable parameters?
          </button>
          <button className="quick-ask-btn" onClick={() => handleSend("Compare BERT and GPT architectures")}>
            Compare BERT and GPT architectures
          </button>
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="main-area">
        {messages.length === 0 ? (
          <div className="hero-section">
            <div className="orb-container">
              <div className="orb-glow"></div>
              <DataSphere />
            </div>
            
            <div className="greeting-text">
              <h1>{greeting}</h1>
              <h2>Can I help you with anything?</h2>
            </div>
          </div>
        ) : (
          <div className="chat-scroll-area">
            {messages.map((msg) => (
              <div key={msg.id} className={`chat-message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'assistant' ? <Bot size={18} className="text-[#8a63f7]" /> : <User size={18} />}
                </div>
                <div className="message-bubble">
                  {msg.role === 'assistant' ? (
                    <div className="markdown">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <div>{msg.content}</div>
                  )}

                  {msg.role === "assistant" && msg.sources && msg.sources.length > 0 && (
                    <div className="mt-4 flex flex-col gap-2">
                      <div className="flex items-center gap-2 text-xs font-semibold text-[var(--text-secondary)] uppercase">
                        <FileText size={14} /> Citations
                      </div>
                      {msg.sources.map((src, idx) => (
                        <div key={idx} className="bg-[rgba(255,255,255,0.03)] border border-[var(--border-light)] rounded-lg p-3 text-sm">
                          <div className="text-[var(--text-primary)] font-medium mb-1">
                            {src.metadata.citation || src.metadata.title}
                          </div>
                          {src.metadata.url && (
                            <a href={src.metadata.url} target="_blank" rel="noreferrer" className="text-[#8a63f7] text-xs flex items-center gap-1 mb-1 hover:underline">
                              ArXiv Link <ExternalLink size={10} />
                            </a>
                          )}
                          <div className="text-[var(--text-secondary)] text-xs italic line-clamp-2">
                            "...{src.text}..."
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && messages[messages.length - 1]?.role === "user" && (
              <div className="chat-message assistant">
                <div className="message-avatar"><Bot size={18} className="text-[#8a63f7]" /></div>
                <div className="message-bubble">
                  <div className="thinking-indicator">
                    <span className="dot"></span>
                    <span className="dot"></span>
                    <span className="dot"></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}

        <div className="bottom-section">
          <div className="input-wrapper">
            <textarea 
              placeholder="Message AI Chat..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />
            <div className="input-actions">
              <div className="action-chips">
                <div className="action-chip" onClick={() => setInput("What is the key innovation in the Transformer?")}>
                  <ImageIcon size={14} /> Compare Models
                </div>
                <div className="action-chip" onClick={() => setInput("Which papers discuss LoRA fine-tuning?")}>
                  <Globe size={14} /> Find Citations
                </div>
              </div>
              <div className="right-actions">
                <button 
                  className="icon-btn primary ml-2" 
                  onClick={() => handleSend()}
                  disabled={!input.trim() || isLoading}
                >
                  <Send size={18} />
                </button>
              </div>
            </div>
          </div>

          {messages.length === 0 && (
            <div className="cards-row">
              <div className="feature-card" onClick={() => handleSend("Explain how LoRA fine-tuning works.")}>
                <h3>Summarize Paper</h3>
                <p>Quickly understand complex ML architectures and techniques.</p>
              </div>
              <div className="feature-card" onClick={() => handleSend("What are the differences between BERT and GPT?")}>
                <h3>Model Analytics</h3>
                <p>Compare performance, parameter efficiency, and use cases.</p>
              </div>
              <div className="feature-card" onClick={() => handleSend("Find papers discussing Retrieval Augmented Generation (RAG).")}>
                <h3>Literature Search</h3>
                <p>Locate exact citations and implementations in the ArXiv database.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
