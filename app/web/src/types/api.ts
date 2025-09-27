// API types matching the backend Pydantic models

export interface AskRequest {
  question: string;
  top_k?: number;
  case_type?: 'cutaneous' | 'mucocutaneous' | 'visceral' | 'unknown' | null;
  keyword?: string | null;
  any_keywords?: string | null;
  micrograph_only?: boolean;
  micrograph_strict?: boolean;
  images_per_answer?: number;
  // File upload support
  uploaded_files?: File[];
}

// Form data version for multipart uploads
export interface AskFormData {
  question: string;
  top_k?: string;
  case_type?: string;
  keyword?: string;
  any_keywords?: string;
  micrograph_only?: string;
  micrograph_strict?: string;
  images_per_answer?: string;
  files?: FileList;
}

export interface HitInfo {
  rank: number;
  score: number;
  doc_id: string;
  page_index: number;
  image_path?: string | null;
  page_kind?: string | null;
  micrograph_like: boolean;
  keywords: string[];
  text_excerpt?: string | null;
}

export type Evidence = [string, string]; // [span_text, citation]

export interface AskResponse {
  answer: string;
  hits: HitInfo[];
  evidence: Evidence[];
  used_images: string[];
  note?: string | null;
  // File upload related responses
  uploaded_file_info?: UploadedFileInfo[];
  processing_status?: string;
}

// Information about uploaded files
export interface UploadedFileInfo {
  filename: string;
  file_type: 'pdf' | 'image';
  pages_extracted?: number;
  processed_images: string[];
  error?: string;
}

// Chat message types
export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  evidence?: Evidence[];
  hits?: HitInfo[];
  used_images?: string[];
  isLoading?: boolean;
  // File upload support
  uploaded_files?: UploadedFileInfo[];
  has_attachments?: boolean;
}

// Settings types
export interface ChatSettings {
  top_k: number;
  case_type: 'cutaneous' | 'mucocutaneous' | 'visceral' | 'unknown' | null;
  keyword: string;
  any_keywords: string;
  micrograph_only: boolean;
  micrograph_strict: boolean;
  images_per_answer: number;
}

// API error types
export interface ApiError {
  detail: string;
  status?: number;
}

// Health check response
export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  service: string;
  rag_initialized: boolean;
}