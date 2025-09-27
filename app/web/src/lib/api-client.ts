// API client for the Medical RAG Chatbot backend

import { AskRequest, AskResponse, HealthResponse, ApiError } from '../types/api';

// Use relative URLs for API calls (Next.js will proxy them)
const BASE_URL = '';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl = BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Make a request to the API with error handling and timeout
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    timeout: number = 180000 // 3 minutes default timeout
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    config.signal = controller.signal;

    try {
      const response = await fetch(url, config);
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch {
          // If JSON parsing fails, use the default error message
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('Request timeout - the server is taking too long to respond. This may happen during model loading.');
        }
        throw error;
      }
      throw new Error('An unexpected error occurred');
    }
  }

  /**
   * Ask a question to the RAG system (JSON-only, for backward compatibility)
   */
  async ask(request: AskRequest): Promise<AskResponse> {
    return this.request<AskResponse>('/api/ask', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Ask a question with file uploads
   */
  async askWithFiles(request: AskRequest, files?: File[]): Promise<AskResponse> {
    // If no files, use the regular JSON endpoint
    if (!files || files.length === 0) {
      return this.ask(request);
    }

    // Create FormData for multipart upload
    const formData = new FormData();
    
    // Add form fields
    formData.append('question', request.question);
    formData.append('top_k', request.top_k?.toString() || '8');
    if (request.case_type) formData.append('case_type', request.case_type);
    if (request.keyword) formData.append('keyword', request.keyword);
    if (request.any_keywords) formData.append('any_keywords', request.any_keywords);
    formData.append('micrograph_only', (request.micrograph_only || false).toString());
    formData.append('micrograph_strict', (request.micrograph_strict || false).toString());
    formData.append('images_per_answer', (request.images_per_answer || 2).toString());
    
    // Add files
    for (const file of files) {
      formData.append('files', file);
    }

    const url = `${this.baseUrl}/api/ask-with-files`;
    const timeout = 300000; // 5 minutes for file uploads
    
    // Create abort controller for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        // Don't set Content-Type header - let browser set it with boundary
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch {
          // If JSON parsing fails, use the default error message
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('File upload timeout - the analysis is taking too long. Please try with smaller files or simpler questions.');
        }
        if (error.message.includes('fetch')) {
          throw new Error('Connection error - please check if the backend service is running and try again.');
        }
        throw error;
      }
      throw new Error('An unexpected error occurred while uploading files');
    }
  }

  /**
   * Check the health of the backend service
   */
  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/healthz');
  }

  /**
   * Get the URL for an image file
   */
  getImageUrl(imagePath: string): string {
    return `${this.baseUrl}/files/${imagePath}`;
  }
}

// Export a singleton instance
export const apiClient = new ApiClient();

// Export the class for testing
export { ApiClient };