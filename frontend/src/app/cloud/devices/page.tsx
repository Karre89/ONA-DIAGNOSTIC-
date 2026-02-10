'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  AlertTriangle,
  BarChart3,
  Building2,
  HardDrive,
  LayoutDashboard,
  LogOut,
  Map,
  RefreshCw,
  Settings,
  Shield,
  Users,
  Wifi,
  WifiOff,
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { api } from '@/lib/api'

const sidebarItems = [
  { icon: LayoutDashboard, label: 'Overview', href: '/cloud' },
  { icon: Building2, label: 'Organizations', href: '/cloud/organizations' },
  { icon: Map, label: 'Sites', href: '/cloud/sites' },
  { icon: HardDrive, label: 'Devices', href: '/cloud/devices', active: true },
  { icon: BarChart3, label: 'Analytics', href: '/cloud' },
  { icon: Users, label: 'Users', href: '/cloud' },
  { icon: Settings, label: 'Settings', href: '/cloud' },
]

interface Device {
  id: string
  name: string
  site: string
  organization: string
  status: string
  last_heartbeat: string | null
}

export default function DevicesPage() {
  const router = useRouter()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [devices, setDevices] = useState<Device[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    if (!api.isAuthenticated()) {
      router.push('/')
      return
    }
    fetchDevices()
  }, [])

  const fetchDevices = async () => {
    try {
      setIsLoading(true)
      const data = await api.getDevices()
      setDevices(data)
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setIsLoading(false)
    }
  }

  const handleLogout = () => {
    api.logout()
    router.push('/')
  }

  const formatLastSeen = (dateStr: string | null) => {
    if (!dateStr) return 'Never'
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
              <div>
                <h1 className="text-lg font-bold text-white">ONA Cloud</h1>
                <p className="text-xs text-gray-400">Admin Portal</p>
              </div>
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
        <header className="glass border-b border-white/5 px-8 py-4 sticky top-0 z-10">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white">Edge Devices</h2>
              <p className="text-sm text-gray-400">Monitor deployed edge devices</p>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={fetchDevices}
                disabled={isLoading}
                className="p-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                <RefreshCw className={`w-5 h-5 text-gray-400 ${isLoading ? 'animate-spin' : ''}`} />
              </button>
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

          {/* Devices Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {isLoading ? (
              [...Array(3)].map((_, i) => (
                <div key={i} className="glass-card rounded-2xl p-6 animate-pulse">
                  <div className="h-6 bg-white/10 rounded w-3/4 mb-4" />
                  <div className="h-4 bg-white/10 rounded w-1/2 mb-2" />
                  <div className="h-4 bg-white/10 rounded w-1/3" />
                </div>
              ))
            ) : devices.length === 0 ? (
              <div className="col-span-full text-center py-12 text-gray-400">
                <HardDrive className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">No devices yet</p>
                <p className="text-sm mt-1">Devices are registered when organizations are seeded</p>
              </div>
            ) : (
              devices.map((device, index) => (
                <motion.div
                  key={device.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="glass-card rounded-2xl p-6 hover:border-ona-primary/30 transition-all cursor-pointer"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-3 rounded-xl ${
                      device.status === 'ONLINE' ? 'bg-green-500/10' : 'bg-gray-500/10'
                    }`}>
                      {device.status === 'ONLINE' ? (
                        <Wifi className="w-6 h-6 text-green-400" />
                      ) : (
                        <WifiOff className="w-6 h-6 text-gray-400" />
                      )}
                    </div>
                    <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                      device.status === 'ONLINE'
                        ? 'bg-green-500/20 text-green-400'
                        : 'bg-gray-500/20 text-gray-400'
                    }`}>
                      {device.status}
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-1">{device.name}</h3>
                  <p className="text-sm text-gray-400 mb-1">{device.site}</p>
                  <p className="text-sm text-gray-500 mb-4">{device.organization}</p>
                  <div className="text-xs text-gray-500">
                    Last seen: {formatLastSeen(device.last_heartbeat)}
                  </div>
                </motion.div>
              ))
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
