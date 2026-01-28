'use client'

import { useState } from 'react'
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
  Users,
  Wifi,
  WifiOff,
  ChevronRight,
  Scan,
} from 'lucide-react'
import Link from 'next/link'

// Mock data - replace with real API calls
const stats = [
  { label: 'Total Scans Today', value: '24', icon: Scan, trend: '+12%' },
  { label: 'High Risk Cases', value: '3', icon: AlertTriangle, color: 'text-red-400' },
  { label: 'Pending Review', value: '7', icon: Clock, color: 'text-yellow-400' },
  { label: 'Completed', value: '14', icon: CheckCircle, color: 'text-green-400' },
]

const recentResults = [
  { id: 'STU-20260128-2AD4C5', risk: 'HIGH', score: 91, time: '2 min ago' },
  { id: 'STU-20260128-38D547', risk: 'LOW', score: 14, time: '15 min ago' },
  { id: 'STU-20260128-7BC123', risk: 'MEDIUM', score: 45, time: '32 min ago' },
  { id: 'STU-20260128-9DE456', risk: 'LOW', score: 8, time: '1 hour ago' },
]

const getRiskColor = (risk: string) => {
  switch (risk) {
    case 'HIGH': return 'bg-red-500/20 text-red-400 border-red-500/30'
    case 'MEDIUM': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
    case 'LOW': return 'bg-green-500/20 text-green-400 border-green-500/30'
    default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
  }
}

export default function DashboardPage() {
  const [isOnline, setIsOnline] = useState(true)

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

              {/* Settings */}
              <button className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <Settings className="w-5 h-5 text-gray-400" />
              </button>

              {/* Logout */}
              <Link href="/" className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <LogOut className="w-5 h-5 text-gray-400" />
              </Link>
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
          <h2 className="text-2xl font-bold text-white mb-2">Good morning, Dr. Smith</h2>
          <p className="text-gray-400">Here's your TB screening overview for today</p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {stats.map((stat, index) => (
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
              <p className="text-3xl font-bold text-white mb-1">{stat.value}</p>
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
              <Link href="/studies" className="text-ona-primary text-sm hover:underline flex items-center gap-1">
                View all <ChevronRight className="w-4 h-4" />
              </Link>
            </div>

            <div className="space-y-3">
              {recentResults.map((result, index) => (
                <motion.div
                  key={result.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5 + index * 0.1 }}
                  className="flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-colors cursor-pointer group"
                >
                  <div className="flex items-center gap-4">
                    <div className={`px-3 py-1 rounded-lg text-sm font-medium border ${getRiskColor(result.risk)}`}>
                      {result.risk}
                    </div>
                    <div>
                      <p className="text-white font-medium">{result.id}</p>
                      <p className="text-sm text-gray-400">{result.time}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <p className={`text-2xl font-bold ${
                        result.score >= 60 ? 'text-red-400' :
                        result.score >= 30 ? 'text-yellow-400' : 'text-green-400'
                      }`}>{result.score}%</p>
                      <p className="text-xs text-gray-400">TB Score</p>
                    </div>
                    <ChevronRight className="w-5 h-5 text-gray-600 group-hover:text-ona-primary transition-colors" />
                  </div>
                </motion.div>
              ))}
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
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-green-400">Synced</span>
              </div>
              <p className="text-sm text-gray-400">Last sync: 2 minutes ago</p>
              <p className="text-sm text-gray-400">24 results synced today</p>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  )
}
