const API_URL = process.env.NEXT_PUBLIC_CLOUD_API_URL || 'https://ona-diagnostic-production.up.railway.app';

interface LoginResponse {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    email: string;
    full_name: string;
    role: string;
    tenant_id?: string;
    site_id?: string;
  };
}

interface DashboardStats {
  total_scans: number;
  high_risk: number;
  medium_risk: number;
  low_risk: number;
  pending_review: number;
  recent_results: Array<{
    id: string;
    study_id: string;
    risk_bucket: string;
    scores: Record<string, number>;
    created_at: string;
  }>;
}

interface CloudStats {
  total_organizations: number;
  total_sites: number;
  total_devices: number;
  total_scans: number;
  devices_online_percent: number;
  organizations: Array<{
    id: string;
    name: string;
    type: string;
    sites: number;
    devices: number;
    status: string;
  }>;
}

class ApiClient {
  private token: string | null = null;

  constructor() {
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('token');
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...options.headers as Record<string, string>,
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${API_URL}/api/v1${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      if (response.status === 401) {
        this.clearToken();
        if (typeof window !== 'undefined') {
          window.location.href = '/';
        }
        throw new Error('Session expired. Redirecting to login...');
      }
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || 'Request failed');
    }

    return response.json();
  }

  setToken(token: string) {
    this.token = token;
    if (typeof window !== 'undefined') {
      localStorage.setItem('token', token);
    }
  }

  clearToken() {
    this.token = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    }
  }

  getToken() {
    return this.token;
  }

  isAuthenticated() {
    return !!this.token;
  }

  // Auth endpoints
  async login(email: string, password: string): Promise<LoginResponse> {
    const response = await this.request<LoginResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    this.setToken(response.access_token);
    if (typeof window !== 'undefined') {
      localStorage.setItem('user', JSON.stringify(response.user));
    }
    return response;
  }

  async register(email: string, password: string, fullName: string, role: string = 'clinic_user'): Promise<LoginResponse> {
    const response = await this.request<LoginResponse>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({
        email,
        password,
        full_name: fullName,
        role
      }),
    });
    this.setToken(response.access_token);
    if (typeof window !== 'undefined') {
      localStorage.setItem('user', JSON.stringify(response.user));
    }
    return response;
  }

  async getMe() {
    return this.request('/auth/me');
  }

  logout() {
    this.clearToken();
  }

  // Dashboard endpoints
  async getDashboardStats(): Promise<DashboardStats> {
    return this.request<DashboardStats>('/auth/dashboard-stats');
  }

  async getCloudStats(): Promise<CloudStats> {
    return this.request<CloudStats>('/auth/cloud-stats');
  }

  // Results endpoints
  async getRecentResults(limit: number = 50) {
    return this.request(`/results/recent?limit=${limit}`);
  }

  async getResult(id: string): Promise<ScanResult> {
    return this.request<ScanResult>(`/results/${id}`);
  }

  // PDF Reports
  async downloadReport(scanId: string): Promise<void> {
    const headers: Record<string, string> = {};
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }
    const response = await fetch(`${API_URL}/api/v1/reports/${scanId}/pdf`, { headers });
    if (!response.ok) throw new Error('Failed to generate report');
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ONA-Report-${scanId.slice(0, 8)}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  }

  // Referral endpoints
  async lookupReferral(code: string): Promise<ReferralInfo> {
    return this.request<ReferralInfo>(`/referrals/lookup/${code}`);
  }

  async listReferrals(status?: string, limit: number = 50): Promise<ReferralInfo[]> {
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    params.set('limit', String(limit));
    return this.request<ReferralInfo[]>(`/referrals?${params.toString()}`);
  }

  async getReferralStats(): Promise<ReferralStats> {
    return this.request<ReferralStats>('/referrals/stats/summary');
  }

  // Seed data (for testing)
  async seedData() {
    return this.request('/seed', { method: 'POST', body: JSON.stringify({}) });
  }

  // Sites endpoints
  async getSites() {
    return this.request<Array<{
      id: string;
      name: string;
      site_code: string;
      country: string;
      organization: string;
      devices: number;
    }>>('/auth/sites');
  }

  // Devices endpoints
  async getDevices() {
    return this.request<Array<{
      id: string;
      name: string;
      site: string;
      organization: string;
      status: string;
      last_heartbeat: string | null;
    }>>('/auth/devices');
  }

  // User management endpoints
  async getUsers(role?: string, isActive?: boolean) {
    const params = new URLSearchParams();
    if (role) params.set('role', role);
    if (isActive !== undefined) params.set('is_active', String(isActive));
    const qs = params.toString();
    return this.request<UserInfo[]>(`/auth/users${qs ? `?${qs}` : ''}`);
  }

  async createUser(data: { email: string; password: string; full_name: string; role: string; tenant_id?: string; site_id?: string }) {
    return this.request<{ id: string; email: string; message: string }>('/auth/users', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateUser(userId: string, data: { full_name?: string; role?: string; is_active?: boolean; tenant_id?: string; site_id?: string }) {
    return this.request<{ id: string; message: string }>(`/auth/users/${userId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteUser(userId: string) {
    return this.request<{ message: string }>(`/auth/users/${userId}`, {
      method: 'DELETE',
    });
  }

  async changePassword(currentPassword: string, newPassword: string) {
    return this.request<{ message: string }>('/auth/change-password', {
      method: 'POST',
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    });
  }
}

interface UserInfo {
  id: string;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
  tenant_name: string | null;
  site_name: string | null;
  tenant_id: string | null;
  site_id: string | null;
  created_at: string | null;
  last_login: string | null;
}

interface ScanResult {
  id: string;
  study_id: string;
  modality: string;
  model_version: string;
  risk_bucket: string;
  scores: Record<string, number>;
  explanation: string | null;
  inference_time_ms: number | null;
  has_burned_in_text: boolean;
  created_at: string;
}

interface ReferralInfo {
  referral_code: string;
  id: string;
  suspected_condition: string;
  symptoms: string[] | null;
  triage_confidence: number | null;
  urgency: string;
  patient_language: string | null;
  patient_demographics: Record<string, any> | null;
  status: string;
  referred_at: string | null;
  scan_id: string | null;
  ona_result: string | null;
  ona_confidence: number | null;
  outcome: string | null;
  created_at: string | null;
}

interface ReferralStats {
  total: number;
  pending: number;
  arrived: number;
  scanned: number;
  completed: number;
  no_show: number;
  conversion_rate: number;
}

export const api = new ApiClient();
export type { LoginResponse, DashboardStats, CloudStats, UserInfo, ScanResult, ReferralInfo, ReferralStats };
