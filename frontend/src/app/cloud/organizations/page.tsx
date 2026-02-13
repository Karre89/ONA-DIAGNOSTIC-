'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  Building2,
  Globe,
  HardDrive,
  LayoutDashboard,
  LogOut,
  Map,
  Plus,
  RefreshCw,
  Settings,
  Shield,
  Users,
  X,
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { api, CloudStats } from '@/lib/api'

const sidebarItems = [
  { icon: LayoutDashboard, label: 'Overview', href: '/cloud' },
  { icon: Building2, label: 'Organizations', href: '/cloud/organizations', active: true },
  { icon: Map, label: 'Sites', href: '/cloud/sites' },
  { icon: HardDrive, label: 'Devices', href: '/cloud/devices' },
  { icon: BarChart3, label: 'Analytics', href: '/cloud/analytics' },
  { icon: Users, label: 'Users', href: '/cloud/users' },
  { icon: Settings, label: 'Settings', href: '/cloud/settings' },
]

export default function OrganizationsPage() {
  const router = useRouter()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [stats, setStats] = useState<CloudStats | null>(null)
  const [error, setError] = useState('')
  const [showAddModal, setShowAddModal] = useState(false)
  const [newOrg, setNewOrg] = useState({ name: '', type: 'NGO' })

  useEffect(() => {
    if (!api.isAuthenticated()) {
      router.push('/')
      return
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
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setIsLoading(false)
    }
  }

  const handleLogout = () => {
    api.logout()
    router.push('/')
  }

  const handleAddOrg = async () => {
    if (!newOrg.name.trim()) return

    try {
      await api.seedData()
      setShowAddModal(false)
      setNewOrg({ name: '', type: 'NGO' })
      fetchStats()
    } catch (err) {
      setError('Failed to add organization')
    }
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
              <h2 className="text-xl font-bold text-white">Organizations</h2>
              <p className="text-sm text-gray-400">Manage healthcare organizations</p>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={fetchStats}
                disabled={isLoading}
                className="p-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                <RefreshCw className={`w-5 h-5 text-gray-400 ${isLoading ? 'animate-spin' : ''}`} />
              </button>
              <button
                onClick={() => setShowAddModal(true)}
                className="btn-primary flex items-center gap-2"
              >
                <Plus className="w-4 h-4" />
                Add Organization
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

          {/* Organizations Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {isLoading ? (
              [...Array(3)].map((_, i) => (
                <div key={i} className="glass-card rounded-2xl p-6 animate-pulse">
                  <div className="h-6 bg-white/10 rounded w-3/4 mb-4" />
                  <div className="h-4 bg-white/10 rounded w-1/2 mb-2" />
                  <div className="h-4 bg-white/10 rounded w-1/3" />
                </div>
              ))
            ) : stats?.organizations.length === 0 ? (
              <div className="col-span-full text-center py-12 text-gray-400">
                <Building2 className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">No organizations yet</p>
                <p className="text-sm mt-1">Click "Add Organization" to get started</p>
              </div>
            ) : (
              stats?.organizations.map((org, index) => (
                <motion.div
                  key={org.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="glass-card rounded-2xl p-6 hover:border-ona-primary/30 transition-all cursor-pointer"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="p-3 rounded-xl bg-ona-primary/10">
                      <Building2 className="w-6 h-6 text-ona-primary" />
                    </div>
                    <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                      org.status === 'active'
                        ? 'bg-green-500/20 text-green-400'
                        : 'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {org.status}
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{org.name}</h3>
                  <p className="text-sm text-gray-400 capitalize mb-4">{org.type}</p>
                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-1 text-gray-400">
                      <Map className="w-4 h-4" />
                      {org.sites} sites
                    </div>
                    <div className="flex items-center gap-1 text-gray-400">
                      <HardDrive className="w-4 h-4" />
                      {org.devices} devices
                    </div>
                  </div>
                </motion.div>
              ))
            )}
          </div>
        </div>
      </main>

      {/* Add Organization Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-card rounded-2xl p-6 w-full max-w-md mx-4"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">Add Organization</h3>
              <button
                onClick={() => setShowAddModal(false)}
                className="p-1 rounded-lg hover:bg-white/10"
              >
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Organization Name</label>
                <input
                  type="text"
                  value={newOrg.name}
                  onChange={(e) => setNewOrg({ ...newOrg, name: e.target.value })}
                  className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:border-ona-primary"
                  placeholder="e.g., Kenyatta National Hospital"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-2">Type</label>
                <select
                  value={newOrg.type}
                  onChange={(e) => setNewOrg({ ...newOrg, type: e.target.value })}
                  className="w-full px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-white focus:outline-none focus:border-ona-primary"
                >
                  <option value="NGO">NGO</option>
                  <option value="Hospital">Hospital</option>
                  <option value="Clinic">Clinic</option>
                  <option value="Government">Government</option>
                </select>
              </div>
              <button
                onClick={handleAddOrg}
                className="w-full btn-primary mt-4"
              >
                Add Organization
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}
