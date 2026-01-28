'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  FileText,
  LogOut,
  Settings,
  TrendingUp,
  Wifi,
  WifiOff,
  ChevronRight,
  Scan,
  RefreshCw,
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { api, DashboardStats } from '@/lib/api'

const getRiskColor = (risk: string) => {
  switch (risk) {
    case 'HIGH': return 'bg-red-500/20 text-red-400 border-red-500/30'
    case 'MEDIUM': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
    case 'LOW': return 'bg-green-500/20 text-green-400 border-green-500/30'
    default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
  }
}

export default function DashboardPage() {
  const router = useRouter()
  const [isOnline, setIsOnline] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [error, setError] = useState('')
  const [user, setUser] = useState<any>(null)

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token')
    const userStr = localStorage.getItem('user')

    if (!token) {
      router.push('/')
      return
    }

    if (userStr) {
      setUser(JSON.parse(userStr))
    }

    // Fetch dashboard stats
    fetchStats()

    // Check online status
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    setIsOnline(navigator.onLine)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [router])

  const fetchStats = async () => {
    try {
      setIsLoading(true)
      const data = await api.getDashboardStats()
      setStats(data)
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load stats')
      // Use fallback data
      setStats({
        total_scans: 0,
        high_risk: 0,
        medium_risk: 0,
        low_risk: 0,
        pending_review: 0,
        recent_results: []
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleLogout = () => {
    api.logout()
    router.push('/')
  }

  const formatTimeAgo = (dateStr: string) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)

    if (days > 0) return `${days}d ago`
    if (hours > 0) return `${hours}h ago`
    if (minutes > 0) return `${minutes}m ago`
    return 'Just now'
  }

  const statsDisplay = [
    { label: 'Total Scans', value: stats?.total_scans || 0, icon: Scan, trend: null },
    { label: 'High Risk Cases', value: stats?.high_risk || 0, icon: AlertTriangle, color: 'text-red-400' },
    { label: 'Pending Review', value: stats?.pending_review || 0, icon: Clock, color: 'text-yellow-400' },
    { label: 'Low Risk', value: stats?.low_risk || 0, icon: CheckCircle, color: 'text-green-400' },
  ]

  return (
    <div className="min-h-screen">
      {/* Navbar */}
      <nav className="glass-strong border-b border-white/5 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-ona-primary to-ona-secondary flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white">ONA Health</h1>
                <p className="text-xs text-gray-400">Edge Device</p>
              </div>
            </div>

            {/* Status & Actions */}
            <div className="flex items-center gap-4">
              {/* Connection Status */}
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                isOnline
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-red-500/20 text-red-400'
              }`}>
                {isOnline ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
                {isOnline ? 'Online' : 'Offline'}
              </div>

              {/* Refresh */}
              <button
                onClick={fetchStats}
                disabled={isLoading}
                className="p-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                <RefreshCw className={`w-5 h-5 text-gray-400 ${isLoading ? 'animate-spin' : ''}`} />
              </button>

              {/* Settings */}
              <button className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <Settings className="w-5 h-5 text-gray-400" />
              </button>

              {/* Logout */}
              <button onClick={handleLogout} className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <LogOut className="w-5 h-5 text-gray-400" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h2 className="text-2xl font-bold text-white mb-2">
            Welcome back, {user?.full_name || 'User'}
          </h2>
          <p className="text-gray-400">Here's your TB screening overview</p>
        </motion.div>

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

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {statsDisplay.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="glass-card rounded-2xl p-5"
            >
              <div className="flex items-start justify-between mb-3">
                <div className={`p-2 rounded-xl bg-ona-primary/10 ${stat.color || 'text-ona-primary'}`}>
                  <stat.icon className="w-5 h-5" />
                </div>
                {stat.trend && (
                  <span className="flex items-center gap-1 text-green-400 text-sm">
                    <TrendingUp className="w-4 h-4" />
                    {stat.trend}
                  </span>
                )}
              </div>
              <p className="text-3xl font-bold text-white mb-1">
                {isLoading ? '-' : stat.value}
              </p>
              <p className="text-sm text-gray-400">{stat.label}</p>
            </motion.div>
          ))}
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Results */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="lg:col-span-2 glass-card rounded-2xl p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">Recent Results</h3>
              <Link href="#" className="text-ona-primary text-sm hover:underline flex items-center gap-1">
                View all <ChevronRight className="w-4 h-4" />
              </Link>
            </div>

            <div className="space-y-3">
              {isLoading ? (
                // Loading skeleton
                [...Array(3)].map((_, i) => (
                  <div key={i} className="p-4 rounded-xl bg-white/5 animate-pulse">
                    <div className="flex items-center gap-4">
                      <div className="w-16 h-8 bg-white/10 rounded-lg" />
                      <div className="flex-1">
                        <div className="h-4 bg-white/10 rounded w-1/3 mb-2" />
                        <div className="h-3 bg-white/10 rounded w-1/4" />
                      </div>
                    </div>
                  </div>
                ))
              ) : stats?.recent_results.length === 0 ? (
                <div className="text-center py-8 text-gray-400">
                  <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No scan results yet</p>
                  <p className="text-sm mt-1">Upload a chest X-ray to get started</p>
                </div>
              ) : (
                stats?.recent_results.map((result, index) => (
                  <motion.div
                    key={result.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 + index * 0.1 }}
                    className="flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-colors cursor-pointer group"
                  >
                    <div className="flex items-center gap-4">
                      <div className={`px-3 py-1 rounded-lg text-sm font-medium border ${getRiskColor(result.risk_bucket)}`}>
                        {result.risk_bucket}
                      </div>
                      <div>
                        <p className="text-white font-medium">Study {result.study_id.slice(0, 12)}</p>
                        <p className="text-sm text-gray-400">{formatTimeAgo(result.created_at)}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <p className={`text-2xl font-bold ${
                          (result.scores?.tb || 0) >= 0.6 ? 'text-red-400' :
                          (result.scores?.tb || 0) >= 0.3 ? 'text-yellow-400' : 'text-green-400'
                        }`}>{Math.round((result.scores?.tb || 0) * 100)}%</p>
                        <p className="text-xs text-gray-400">TB Score</p>
                      </div>
                      <ChevronRight className="w-5 h-5 text-gray-600 group-hover:text-ona-primary transition-colors" />
                    </div>
                  </motion.div>
                ))
              )}
            </div>
          </motion.div>

          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="space-y-4"
          >
            {/* New Scan */}
            <div className="glass-card rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button className="btn-primary w-full flex items-center justify-center gap-2">
                  <Scan className="w-5 h-5" />
                  New Scan
                </button>
                <button className="w-full glass p-3 rounded-xl text-gray-300 hover:text-white hover:border-ona-primary/50 transition-all flex items-center justify-center gap-2">
                  <FileText className="w-5 h-5" />
                  View Reports
                </button>
              </div>
            </div>

            {/* Model Info */}
            <div className="glass-card rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">AI Model</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Version</span>
                  <span className="text-white font-medium">v2.0</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Architecture</span>
                  <span className="text-white font-medium">ResNet18</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Status</span>
                  <span className="text-green-400 font-medium">Active</span>
                </div>
              </div>
            </div>

            {/* Sync Status */}
            <div className="glass-card rounded-2xl p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Sync Status</h3>
              <div className="flex items-center gap-3 mb-3">
                <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-400' : 'bg-yellow-400'} animate-pulse`} />
                <span className={isOnline ? 'text-green-400' : 'text-yellow-400'}>
                  {isOnline ? 'Synced' : 'Offline Mode'}
                </span>
              </div>
              <p className="text-sm text-gray-400">
                {stats?.total_scans || 0} results in database
              </p>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  )
}
