'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  Building2,
  CheckCircle,
  Clock,
  HardDrive,
  LayoutDashboard,
  LogOut,
  Map,
  Monitor,
  Settings,
  Shield,
  TrendingUp,
  Users,
  Zap,
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { api, CloudStats, DashboardStats, ReferralStats } from '@/lib/api'

const sidebarItems = [
  { icon: LayoutDashboard, label: 'Overview', href: '/cloud' },
  { icon: Building2, label: 'Organizations', href: '/cloud/organizations' },
  { icon: Map, label: 'Sites', href: '/cloud/sites' },
  { icon: HardDrive, label: 'Devices', href: '/cloud/devices' },
  { icon: BarChart3, label: 'Analytics', href: '/cloud/analytics', active: true },
  { icon: Users, label: 'Users', href: '/cloud/users' },
  { icon: Settings, label: 'Settings', href: '/cloud/settings' },
]

export default function AnalyticsPage() {
  const router = useRouter()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [cloudStats, setCloudStats] = useState<CloudStats | null>(null)
  const [dashStats, setDashStats] = useState<DashboardStats | null>(null)
  const [referralStats, setReferralStats] = useState<ReferralStats | null>(null)
  const [error, setError] = useState('')
  const [user, setUser] = useState<any>(null)

  useEffect(() => {
    if (!api.isAuthenticated()) {
      router.push('/')
      return
    }
    const userStr = localStorage.getItem('user')
    if (userStr) setUser(JSON.parse(userStr))
    fetchAll()
  }, [])

  const fetchAll = async () => {
    try {
      setIsLoading(true)
      const [cloud, dash, refs] = await Promise.all([
        api.getCloudStats().catch(() => null),
        api.getDashboardStats().catch(() => null),
        api.getReferralStats().catch(() => null),
      ])
      if (cloud) setCloudStats(cloud)
      if (dash) setDashStats(dash)
      if (refs) setReferralStats(refs)
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analytics')
    } finally {
      setIsLoading(false)
    }
  }

  const handleLogout = () => {
    api.logout()
    router.push('/')
  }

  // Build risk distribution for bar chart
  const riskData = dashStats ? [
    { label: 'High Risk', count: dashStats.high_risk, color: 'bg-red-500', textColor: 'text-red-400' },
    { label: 'Medium Risk', count: dashStats.medium_risk, color: 'bg-yellow-500', textColor: 'text-yellow-400' },
    { label: 'Low Risk', count: dashStats.low_risk, color: 'bg-green-500', textColor: 'text-green-400' },
  ] : []

  const maxRisk = Math.max(...riskData.map(r => r.count), 1)

  // Referral pipeline stages
  const pipelineStages = referralStats ? [
    { label: 'Pending', count: referralStats.pending, color: 'bg-yellow-500' },
    { label: 'Arrived', count: referralStats.arrived, color: 'bg-blue-500' },
    { label: 'Scanned', count: referralStats.scanned, color: 'bg-purple-500' },
    { label: 'Completed', count: referralStats.completed, color: 'bg-green-500' },
    { label: 'No Show', count: referralStats.no_show, color: 'bg-red-500' },
  ] : []

  const maxPipeline = Math.max(...pipelineStages.map(s => s.count), 1)

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

        <div className="absolute bottom-4 left-4 right-4 space-y-2">
          <Link
            href="/dashboard"
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-ona-primary hover:bg-ona-primary/10 transition-all"
          >
            <Monitor className="w-5 h-5" />
            {sidebarOpen && <span>Clinic View</span>}
          </Link>
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
        <header className="glass border-b border-white/5 px-8 py-4 sticky top-0 z-10">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white">Analytics</h2>
              <p className="text-sm text-gray-400">Platform performance and screening insights</p>
            </div>
            <div className="glass px-4 py-2 rounded-xl flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-ona-primary" />
              <span className="text-sm text-gray-300">{user?.full_name || 'Admin'}</span>
            </div>
          </div>
        </header>

        <div className="p-8">
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

          {/* Top KPIs */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {[
              { label: 'Total Scans', value: dashStats?.total_scans ?? '-', icon: Activity, color: 'text-ona-primary' },
              { label: 'Devices Online', value: cloudStats ? `${cloudStats.devices_online_percent}%` : '-', icon: HardDrive, color: 'text-green-400' },
              { label: 'Active Sites', value: cloudStats?.total_sites ?? '-', icon: Map, color: 'text-blue-400' },
              { label: 'Referral Conversion', value: referralStats ? `${referralStats.conversion_rate}%` : '-', icon: TrendingUp, color: 'text-purple-400' },
            ].map((kpi, index) => (
              <motion.div
                key={kpi.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="glass-card rounded-2xl p-5"
              >
                <div className={`p-2 rounded-xl bg-white/5 ${kpi.color} w-fit mb-3`}>
                  <kpi.icon className="w-5 h-5" />
                </div>
                <p className="text-3xl font-bold text-white mb-1">{isLoading ? '-' : kpi.value}</p>
                <p className="text-sm text-gray-400">{kpi.label}</p>
              </motion.div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* Risk Distribution */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-card rounded-2xl p-6"
            >
              <h3 className="text-lg font-semibold text-white mb-6">Risk Distribution</h3>
              {isLoading ? (
                <div className="space-y-4">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="h-10 bg-white/5 rounded-lg animate-pulse" />
                  ))}
                </div>
              ) : dashStats?.total_scans === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <BarChart3 className="w-10 h-10 mx-auto mb-2 opacity-40" />
                  <p>No scan data yet</p>
                </div>
              ) : (
                <div className="space-y-5">
                  {riskData.map((item, index) => (
                    <motion.div
                      key={item.label}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.4 + index * 0.1 }}
                    >
                      <div className="flex justify-between mb-2">
                        <span className="text-sm text-gray-300">{item.label}</span>
                        <span className={`text-sm font-bold ${item.textColor}`}>{item.count}</span>
                      </div>
                      <div className="h-8 bg-white/5 rounded-lg overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${(item.count / maxRisk) * 100}%` }}
                          transition={{ delay: 0.5 + index * 0.1, duration: 0.6 }}
                          className={`h-full rounded-lg ${item.color}`}
                          style={{ opacity: 0.6, minWidth: item.count > 0 ? '2rem' : 0 }}
                        />
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </motion.div>

            {/* Device Health */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="glass-card rounded-2xl p-6"
            >
              <h3 className="text-lg font-semibold text-white mb-6">Platform Health</h3>
              {isLoading ? (
                <div className="space-y-4">
                  {[...Array(4)].map((_, i) => (
                    <div key={i} className="h-8 bg-white/5 rounded-lg animate-pulse" />
                  ))}
                </div>
              ) : (
                <div className="space-y-6">
                  {/* Device uptime ring */}
                  <div className="flex items-center gap-6">
                    <div className="relative w-24 h-24">
                      <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 96 96">
                        <circle cx="48" cy="48" r="40" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
                        <circle
                          cx="48" cy="48" r="40" fill="none"
                          stroke="url(#gradient)" strokeWidth="8"
                          strokeLinecap="round"
                          strokeDasharray={`${(cloudStats?.devices_online_percent || 0) * 2.51} 251`}
                        />
                        <defs>
                          <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#00CED1" />
                            <stop offset="100%" stopColor="#20B2AA" />
                          </linearGradient>
                        </defs>
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-xl font-bold text-white">{cloudStats?.devices_online_percent || 0}%</span>
                      </div>
                    </div>
                    <div>
                      <p className="text-white font-medium">Devices Online</p>
                      <p className="text-sm text-gray-400">{cloudStats?.total_devices || 0} total devices</p>
                    </div>
                  </div>

                  <div className="divider-glow" />

                  {/* Stats rows */}
                  <div className="space-y-4">
                    {[
                      { label: 'Organizations', value: cloudStats?.total_organizations || 0, icon: Building2 },
                      { label: 'Active Sites', value: cloudStats?.total_sites || 0, icon: Map },
                      { label: 'Total Scans', value: cloudStats?.total_scans || 0, icon: Activity },
                      { label: 'Pending Review', value: dashStats?.pending_review || 0, icon: Clock },
                    ].map((row) => (
                      <div key={row.label} className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <row.icon className="w-4 h-4 text-gray-500" />
                          <span className="text-sm text-gray-400">{row.label}</span>
                        </div>
                        <span className="text-white font-medium">{row.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          </div>

          {/* Referral Pipeline */}
          {referralStats && referralStats.total > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="glass-card rounded-2xl p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-white">SYNARA Referral Pipeline</h3>
                <span className="text-sm text-gray-400">{referralStats.total} total referrals</span>
              </div>

              <div className="grid grid-cols-5 gap-4">
                {pipelineStages.map((stage, index) => (
                  <motion.div
                    key={stage.label}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 + index * 0.1 }}
                    className="text-center"
                  >
                    <div className="h-32 flex items-end justify-center mb-3">
                      <motion.div
                        initial={{ height: 0 }}
                        animate={{ height: `${Math.max((stage.count / maxPipeline) * 100, 8)}%` }}
                        transition={{ delay: 0.7 + index * 0.1, duration: 0.5 }}
                        className={`w-12 rounded-t-lg ${stage.color}`}
                        style={{ opacity: 0.6 }}
                      />
                    </div>
                    <p className="text-xl font-bold text-white">{stage.count}</p>
                    <p className="text-xs text-gray-400 mt-1">{stage.label}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </main>
    </div>
  )
}
