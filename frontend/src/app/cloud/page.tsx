'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Building2,
  Globe,
  HardDrive,
  LayoutDashboard,
  LogOut,
  Map,
  RefreshCw,
  Settings,
  Shield,
  Users,
  Zap,
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { api, CloudStats } from '@/lib/api'

const sidebarItems = [
  { icon: LayoutDashboard, label: 'Overview', href: '/cloud', active: true },
  { icon: Building2, label: 'Organizations', href: '/cloud/organizations' },
  { icon: Map, label: 'Sites', href: '/cloud/sites' },
  { icon: HardDrive, label: 'Devices', href: '/cloud/devices' },
  { icon: BarChart3, label: 'Analytics', href: '/cloud' },
  { icon: Users, label: 'Users', href: '/cloud' },
  { icon: Settings, label: 'Settings', href: '/cloud' },
]

export default function CloudDashboardPage() {
  const router = useRouter()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [stats, setStats] = useState<CloudStats | null>(null)
  const [error, setError] = useState('')
  const [user, setUser] = useState<any>(null)

  useEffect(() => {
    // Load user info if available
    const userStr = localStorage.getItem('user')
    if (userStr) {
      setUser(JSON.parse(userStr))
    }

    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      setIsLoading(true)
      const data = await api.getCloudStats()
      setStats(data)
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load stats')
    } finally {
      setIsLoading(false)
    }
  }

  const handleLogout = () => {
    api.logout()
    router.push('/')
  }

  const orgStats = [
    { label: 'Total Organizations', value: stats?.total_organizations || 0, icon: Building2 },
    { label: 'Active Sites', value: stats?.total_sites || 0, icon: Map },
    { label: 'Edge Devices', value: stats?.total_devices || 0, icon: HardDrive },
    { label: 'Total Scans', value: stats?.total_scans || 0, icon: Activity },
  ]

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className={`glass-strong border-r border-white/5 transition-all duration-300 ${
        sidebarOpen ? 'w-64' : 'w-20'
      }`}>
        <div className="p-4 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-ona-primary to-ona-secondary flex items-center justify-center flex-shrink-0">
              <Shield className="w-5 h-5 text-white" />
            </div>
            {sidebarOpen && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <h1 className="text-lg font-bold text-white">ONA Cloud</h1>
                <p className="text-xs text-gray-400">Admin Portal</p>
              </motion.div>
            )}
          </div>
        </div>

        <nav className="p-4 space-y-2">
          {sidebarItems.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all ${
                item.active
                  ? 'bg-ona-primary/20 text-ona-primary'
                  : 'text-gray-400 hover:bg-white/5 hover:text-white'
              }`}
            >
              <item.icon className="w-5 h-5 flex-shrink-0" />
              {sidebarOpen && <span>{item.label}</span>}
            </Link>
          ))}
        </nav>

        <div className="absolute bottom-4 left-4 right-4">
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-gray-400 hover:bg-white/5 hover:text-white transition-all"
          >
            <LogOut className="w-5 h-5" />
            {sidebarOpen && <span>Sign Out</span>}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {/* Header */}
        <header className="glass border-b border-white/5 px-8 py-4 sticky top-0 z-10">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white">Dashboard Overview</h2>
              <p className="text-sm text-gray-400">Monitor your global TB screening network</p>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={fetchStats}
                disabled={isLoading}
                className="p-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                <RefreshCw className={`w-5 h-5 text-gray-400 ${isLoading ? 'animate-spin' : ''}`} />
              </button>
              <div className="glass px-4 py-2 rounded-xl flex items-center gap-2">
                <Globe className="w-4 h-4 text-ona-primary" />
                <span className="text-sm text-gray-300">{user?.full_name || 'Admin'}</span>
              </div>
            </div>
          </div>
        </header>

        <div className="p-8">
          {/* Error Banner */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 flex items-center gap-3"
            >
              <AlertTriangle className="w-5 h-5" />
              <span>{error}</span>
            </motion.div>
          )}

          {/* Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {orgStats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="glass-card rounded-2xl p-5"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="p-2 rounded-xl bg-ona-primary/10 text-ona-primary">
                    <stat.icon className="w-5 h-5" />
                  </div>
                </div>
                <p className="text-3xl font-bold text-white mb-1">{isLoading ? '-' : stat.value}</p>
                <p className="text-sm text-gray-400">{stat.label}</p>
              </motion.div>
            ))}
          </div>

          {/* Main Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Organizations Table */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="lg:col-span-2 glass-card rounded-2xl p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white">Organizations</h3>
                <button className="btn-primary text-sm py-2 px-4">Add Organization</button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Organization</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Type</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Sites</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Devices</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {isLoading ? (
                      [...Array(3)].map((_, i) => (
                        <tr key={i} className="border-b border-white/5">
                          <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-32 animate-pulse" /></td>
                          <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-20 animate-pulse" /></td>
                          <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-8 animate-pulse" /></td>
                          <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-8 animate-pulse" /></td>
                          <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-16 animate-pulse" /></td>
                        </tr>
                      ))
                    ) : stats?.organizations.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="py-8 text-center text-gray-400">
                          No organizations yet. Add one to get started.
                        </td>
                      </tr>
                    ) : (
                      stats?.organizations.map((org, index) => (
                        <motion.tr
                          key={org.id}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.5 + index * 0.1 }}
                          className="border-b border-white/5 hover:bg-white/5 cursor-pointer"
                        >
                          <td className="py-4 px-4">
                            <p className="text-white font-medium">{org.name}</p>
                          </td>
                          <td className="py-4 px-4 text-gray-400 capitalize">{org.type}</td>
                          <td className="py-4 px-4 text-gray-400">{org.sites}</td>
                          <td className="py-4 px-4 text-gray-400">{org.devices}</td>
                          <td className="py-4 px-4">
                            <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                              org.status === 'active'
                                ? 'bg-green-500/20 text-green-400'
                                : 'bg-yellow-500/20 text-yellow-400'
                            }`}>
                              {org.status}
                            </span>
                          </td>
                        </motion.tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </motion.div>

            {/* Quick Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="space-y-4"
            >
              {/* Platform Health */}
              <div className="glass-card rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Platform Health</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-gray-400 text-sm">Devices Online</span>
                      <span className="text-green-400 text-sm">{isLoading ? '-' : `${stats?.devices_online_percent || 0}%`}</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-ona-primary to-ona-secondary rounded-full transition-all"
                        style={{ width: `${stats?.devices_online_percent || 0}%` }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-gray-400 text-sm">Sync Success Rate</span>
                      <span className="text-green-400 text-sm">99.7%</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-ona-primary to-ona-secondary w-[99.7%] rounded-full" />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-gray-400 text-sm">API Uptime</span>
                      <span className="text-green-400 text-sm">99.99%</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-ona-primary to-ona-secondary w-[99.99%] rounded-full" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="glass-card rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
                <div className="space-y-2">
                  <button className="w-full glass p-3 rounded-xl text-gray-300 hover:text-white hover:border-ona-primary/50 transition-all flex items-center gap-3">
                    <Zap className="w-5 h-5 text-ona-primary" />
                    Push Model Update
                  </button>
                  <button className="w-full glass p-3 rounded-xl text-gray-300 hover:text-white hover:border-ona-primary/50 transition-all flex items-center gap-3">
                    <BarChart3 className="w-5 h-5 text-ona-primary" />
                    Generate Report
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </main>
    </div>
  )
}
