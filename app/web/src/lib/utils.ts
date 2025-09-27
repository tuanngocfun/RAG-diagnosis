// Utility functions for the chat interface

/**
 * Combine class names (simple version without clsx)
 */
export function cn(...inputs: (string | undefined | null | boolean)[]) {
  return inputs.filter(Boolean).join(' ');
}

/**
 * Format a timestamp for display
 */
export function formatTimestamp(date: Date): string {
  return new Intl.DateTimeFormat('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  }).format(date);
}

/**
 * Format a score for display (2 decimal places)
 */
export function formatScore(score: number): string {
  return score.toFixed(2);
}

/**
 * Truncate text to a maximum length
 */
export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength).trim() + '...';
}

/**
 * Generate a unique ID
 */
export function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

/**
 * Parse citations from text and add superscript links
 */
export function parseCitations(text: string): string {
  // Replace [1], [2], etc. with superscript spans
  return text.replace(/\[(\d+)\]/g, '<sup class="citation">[$1]</sup>');
}

/**
 * Debounce function for search inputs
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: any;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
}

/**
 * Check if a string is a valid case type
 */
export function isValidCaseType(value: string): value is 'cutaneous' | 'mucocutaneous' | 'visceral' | 'unknown' {
  const validTypes = ['cutaneous', 'mucocutaneous', 'visceral', 'unknown'];
  return validTypes.indexOf(value) !== -1;
}

/**
 * Get a color class based on case type
 */
export function getCaseTypeColor(caseType: string): string {
  switch (caseType) {
    case 'cutaneous':
      return 'text-blue-600 bg-blue-50 border-blue-200';
    case 'mucocutaneous':
      return 'text-purple-600 bg-purple-50 border-purple-200';
    case 'visceral':
      return 'text-red-600 bg-red-50 border-red-200';
    case 'unknown':
      return 'text-gray-600 bg-gray-50 border-gray-200';
    default:
      return 'text-gray-600 bg-gray-50 border-gray-200';
  }
}

/**
 * Get an icon for page kind
 */
export function getPageKindIcon(pageKind: string): string {
  switch (pageKind) {
    case 'figure_or_micrograph':
      return '🔬';
    case 'text_heavy':
      return '📄';
    case 'mixed':
      return '📊';
    default:
      return '📑';
  }
}

/**
 * Format keywords for display
 */
export function formatKeywords(keywords: string[]): string {
  if (keywords.length === 0) return '';
  if (keywords.length <= 3) return keywords.join(', ');
  return `${keywords.slice(0, 3).join(', ')} +${keywords.length - 3}`;
}

/**
 * Copy text to clipboard
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    const success = document.execCommand('copy');
    document.body.removeChild(textArea);
    return success;
  }
}