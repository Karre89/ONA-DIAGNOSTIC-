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

  async getResult(id: string) {
    return this.request(`/results/${id}`);
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
}

export const api = new ApiClient();
export type { LoginResponse, DashboardStats, CloudStats };
