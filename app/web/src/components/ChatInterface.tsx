'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, ChatSettings, AskRequest } from '../types/api';
import { apiClient } from '../lib/api-client';
import { generateId, formatTimestamp, formatScore, formatKeywords } from '../lib/utils';
import FileUpload from './FileUpload';

// Simple icon components to replace lucide-react
const Send = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
  </svg>
);

const Settings = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

const ChevronDown = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
  </svg>
);

const ChevronUp = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
  </svg>
);

const Paperclip = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
  </svg>
);

const Copy = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
);

const DEFAULT_SETTINGS: ChatSettings = {
  top_k: 8,
  case_type: null,
  keyword: '',
  any_keywords: '',
  micrograph_only: false,
  micrograph_strict: false,
  images_per_answer: 2,
};

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [showSettings, setShowSettings] = useState(false);
  const [expandedEvidence, setExpandedEvidence] = useState<string | null>(null);
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const hasAttachments = selectedFiles.length > 0;
    
    const userMessage: ChatMessage = {
      id: generateId(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
      has_attachments: hasAttachments,
    };

    const loadingMessage: ChatMessage = {
      id: generateId(),
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      isLoading: true,
    };

    setMessages(prev => [...prev, userMessage, loadingMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const request: AskRequest = {
        question: inputValue,
        top_k: settings.top_k,
        case_type: settings.case_type,
        keyword: settings.keyword || null,
        any_keywords: settings.any_keywords || null,
        micrograph_only: settings.micrograph_only,
        micrograph_strict: settings.micrograph_strict,
        images_per_answer: settings.images_per_answer,
      };

      // Use the appropriate API call based on whether files are attached
      const response = await apiClient.askWithFiles(request, selectedFiles.length > 0 ? selectedFiles : undefined);

      const assistantMessage: ChatMessage = {
        id: loadingMessage.id,
        type: 'assistant',
        content: response.answer || (response.note || 'No answer available'),
        timestamp: new Date(),
        evidence: response.evidence,
        hits: response.hits,
        used_images: response.used_images,
        uploaded_files: response.uploaded_file_info,
      };

      setMessages(prev => 
        prev.map(msg => 
          msg.id === loadingMessage.id ? assistantMessage : msg
        )
      );
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: loadingMessage.id,
        type: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };

      setMessages(prev => 
        prev.map(msg => 
          msg.id === loadingMessage.id ? errorMessage : msg
        )
      );
    } finally {
      setIsLoading(false);
      // Clear uploaded files after submission
      setSelectedFiles([]);
      setShowFileUpload(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setExpandedEvidence(null);
    setSelectedFiles([]);
    setShowFileUpload(false);
  };

  const handleFilesChange = (files: File[]) => {
    setSelectedFiles(files);
  };

  const copyMessage = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const toggleEvidence = (messageId: string) => {
    setExpandedEvidence(prev => prev === messageId ? null : messageId);
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Settings Sidebar */}
      <div className={`${showSettings ? 'w-80' : 'w-0'} transition-all duration-300 overflow-hidden bg-white border-r border-gray-200`}>
        <div className="p-6">
          <h2 className="text-lg font-semibold mb-4">Settings</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Top K Results
              </label>
              <input
                type="number"
                min="1"
                max="15"
                value={settings.top_k}
                onChange={(e) => setSettings(prev => ({ ...prev, top_k: parseInt(e.target.value) || 8 }))}
                className="input-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Case Type
              </label>
              <select
                value={settings.case_type || ''}
                onChange={(e) => setSettings(prev => ({ ...prev, case_type: e.target.value as any || null }))}
                className="input-primary"
              >
                <option value="">All Types</option>
                <option value="cutaneous">Cutaneous</option>
                <option value="mucocutaneous">Mucocutaneous</option>
                <option value="visceral">Visceral</option>
                <option value="unknown">Unknown</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Keyword Filter
              </label>
              <input
                type="text"
                value={settings.keyword}
                onChange={(e) => setSettings(prev => ({ ...prev, keyword: e.target.value }))}
                placeholder="e.g., microscopy"
                className="input-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Any Keywords (comma-separated)
              </label>
              <input
                type="text"
                value={settings.any_keywords}
                onChange={(e) => setSettings(prev => ({ ...prev, any_keywords: e.target.value }))}
                placeholder="e.g., amastigotes, diagnosis, PCR"
                className="input-primary"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Images per Answer
              </label>
              <select
                value={settings.images_per_answer}
                onChange={(e) => setSettings(prev => ({ ...prev, images_per_answer: parseInt(e.target.value) }))}
                className="input-primary"
              >
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
                <option value={4}>4</option>
                <option value={5}>5</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={settings.micrograph_only}
                  onChange={(e) => setSettings(prev => ({ ...prev, micrograph_only: e.target.checked }))}
                  className="rounded border-gray-300 text-medical-600 focus:ring-medical-500"
                />
                <span className="ml-2 text-sm text-gray-700">Prefer micrographs</span>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={settings.micrograph_strict}
                  onChange={(e) => setSettings(prev => ({ ...prev, micrograph_strict: e.target.checked }))}
                  className="rounded border-gray-300 text-medical-600 focus:ring-medical-500"
                />
                <span className="ml-2 text-sm text-gray-700">Micrographs only</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-900">Medical RAG Chatbot</h1>
            <p className="text-sm text-gray-600">Ask questions about Leishmania cases</p>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="btn-ghost"
            >
              <Settings className="w-4 h-4" />
            </button>
            
            <button
              onClick={clearChat}
              className="btn-secondary"
            >
              Clear Chat
            </button>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="bg-yellow-50 border-b border-yellow-200 px-6 py-3">
          <p className="text-sm text-yellow-800">
            ⚠️ <strong>For educational use only.</strong> This tool is not intended for medical diagnosis or treatment decisions.
          </p>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-gray-500 mt-8">
              <h2 className="text-lg font-medium mb-2">Welcome to the Medical RAG Chatbot</h2>
              <p>Ask questions about Leishmania cases, diagnostic features, treatments, and more.</p>
              <div className="mt-4 text-sm text-gray-400">
                <p>Example questions:</p>
                <ul className="mt-2 space-y-1">
                  <li>• "What are the diagnostic features on biopsy?"</li>
                  <li>• "Show me microscopy findings of amastigotes"</li>
                  <li>• "What treatments are recommended for cutaneous leishmaniasis?"</li>
                </ul>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className="animate-fade-in">
              <div className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-4xl responsive-container ${message.type === 'user' ? 'message-user' : 'message-assistant'}`}>
                  {message.isLoading ? (
                    <div className="flex items-center space-x-2">
                      <div className="loading-dots">
                        <span style={{ '--delay': '0' } as any}></span>
                        <span style={{ '--delay': '1' } as any}></span>
                        <span style={{ '--delay': '2' } as any}></span>
                      </div>
                      <span className="text-sm text-gray-500">Thinking...</span>
                    </div>
                  ) : (
                    <>
                        <div className="prose prose-sm max-w-none break-words preserve-whitespace">
                        <div dangerouslySetInnerHTML={{ __html: message.content.replace(/\[(\d+)\]/g, '<sup class="citation">[$1]</sup>') }} />
                      </div>
                      
                      {/* Show attachment indicator for user messages */}
                      {message.type === 'user' && message.has_attachments && (
                        <div className="mt-2 text-xs text-gray-500 flex items-center">
                          <Paperclip className="w-3 h-3 mr-1" />
                          <span>Attached files</span>
                        </div>
                      )}
                      
                      {/* Show upload results for assistant messages */}
                      {message.type === 'assistant' && message.uploaded_files && message.uploaded_files.length > 0 && (
                        <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-md">
                          <h5 className="text-sm font-medium text-blue-900 mb-2">Processed Files:</h5>
                          <div className="space-y-1">
                            {message.uploaded_files.map((file, idx) => (
                              <div key={idx} className="text-sm text-blue-800">
                                📁 {file.filename} ({file.file_type}
                                {file.pages_extracted && `, ${file.pages_extracted} pages`})
                                {file.error && <span className="text-red-600"> - Error: {file.error}</span>}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}                      {message.type === 'assistant' && (
                        <div className="mt-3 flex items-center space-x-2 text-xs text-gray-500">
                          <span>{formatTimestamp(message.timestamp)}</span>
                          <button
                            onClick={() => copyMessage(message.content)}
                            className="hover:text-gray-700"
                          >
                            <Copy className="w-3 h-3" />
                          </button>
                          {(message.evidence?.length || 0) > 0 && (
                            <button
                              onClick={() => toggleEvidence(message.id)}
                              className="hover:text-gray-700 flex items-center space-x-1"
                            >
                              <span>Show Evidence</span>
                              {expandedEvidence === message.id ? 
                                <ChevronUp className="w-3 h-3" /> : 
                                <ChevronDown className="w-3 h-3" />
                              }
                            </button>
                          )}
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>

              {/* Evidence Section */}
              {message.type === 'assistant' && expandedEvidence === message.id && (
                <div className="mt-4 bg-gray-50 border border-gray-200 rounded-lg overflow-hidden w-full">
                  {/* Sources */}
                  {message.evidence && message.evidence.length > 0 && (
                    <div className="p-4 border-b border-gray-200">
                      <h4 className="font-medium text-sm text-gray-900 mb-2">Sources</h4>
                      <div className="space-y-2">
                        {message.evidence.map(([span, cite], idx) => (
                          <div key={idx} className="text-sm break-words">
                            <span className="font-medium text-gray-700">[{idx + 1}]</span>
                            <span className="ml-2 text-gray-600 break-words">{span}</span>
                            <span className="ml-2 text-xs text-gray-500 break-words">({cite})</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Retrieved Evidence Table */}
                  {message.hits && message.hits.length > 0 && (
                    <div className="p-4 w-full">
                      <h4 className="font-medium text-sm text-gray-900 mb-3">Retrieved Evidence</h4>
                      <div className="table-responsive w-full">
                        <table className="evidence-table">
                          <thead>
                            <tr>
                              <th>Rank</th>
                              <th>Score</th>
                              <th>Doc ID</th>
                              <th>Page</th>
                              <th>Type</th>
                              <th>Keywords</th>
                              <th>Excerpt</th>
                            </tr>
                          </thead>
                          <tbody>
                            {message.hits.map((hit) => (
                              <tr key={`${hit.doc_id}-${hit.page_index}`}>
                                <td>{hit.rank}</td>
                                <td>{formatScore(hit.score)}</td>
                                <td className="text-xs font-mono break-long-words">{hit.doc_id}</td>
                                <td>{hit.page_index}</td>
                                <td>
                                  <span className="text-xs px-2 py-1 bg-gray-100 rounded whitespace-nowrap">
                                    {hit.micrograph_like ? '🔬' : '📄'} {hit.page_kind}
                                  </span>
                                </td>
                                <td className="text-xs break-words">{formatKeywords(hit.keywords)}</td>
                                <td className="text-xs max-w-xs break-words">{hit.text_excerpt}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Used Images */}
                  {message.used_images && message.used_images.length > 0 && (
                    <div className="p-4 border-t border-gray-200">
                      <h4 className="font-medium text-sm text-gray-900 mb-3">Images Used</h4>
                      <div className="flex space-x-2 overflow-x-auto">
                        {message.used_images.map((imagePath, idx) => (
                          <div key={idx} className="flex-shrink-0">
                            <img
                              src={`/files/${imagePath}`}
                              alt={`Medical image ${idx + 1}`}
                              className="image-thumbnail w-20 h-20"
                              onError={(e) => {
                                (e.target as HTMLImageElement).style.display = 'none';
                              }}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          {/* File Upload Section */}
          {showFileUpload && (
            <div className="mb-4">
              <FileUpload
                onFilesChange={handleFilesChange}
                maxFiles={3}
                maxFileSize={50}
                disabled={isLoading}
              />
            </div>
          )}
          
          <form onSubmit={handleSubmit} className="flex space-x-4">
            <div className="flex-1 flex space-x-2">
              <button
                type="button"
                onClick={() => setShowFileUpload(!showFileUpload)}
                className={`px-3 py-2 rounded-md border transition-colors ${
                  showFileUpload || selectedFiles.length > 0
                    ? 'bg-medical-50 border-medical-300 text-medical-700'
                    : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'
                }`}
                disabled={isLoading}
                title="Attach medical files (PDFs, images)"
              >
                <div className="flex items-center space-x-1">
                  <Paperclip className="w-4 h-4" />
                  {selectedFiles.length > 0 && (
                    <span className="text-xs font-medium bg-medical-100 text-medical-800 px-1.5 py-0.5 rounded-full">
                      {selectedFiles.length}
                    </span>
                  )}
                </div>
              </button>
              
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder={selectedFiles.length > 0 ? 
                  "Ask a question about your uploaded files..." : 
                  "Ask a medical question..."
                }
                disabled={isLoading}
                className="flex-1 input-primary"
              />
            </div>
            
            <button
              type="submit"
              disabled={isLoading || !inputValue.trim()}
              className="btn-primary"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          
          {/* File attachment summary */}
          {selectedFiles.length > 0 && !showFileUpload && (
            <div className="mt-2 text-sm text-gray-600 flex items-center">
              <Paperclip className="w-3 h-3 mr-1" />
              <span>{selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''} attached</span>
              <button
                type="button"
                onClick={() => setShowFileUpload(true)}
                className="ml-2 text-medical-600 hover:text-medical-700 underline"
              >
                View/Edit
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}