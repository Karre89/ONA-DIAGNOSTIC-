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
  MapPin,
  Monitor,
  Plus,
  RefreshCw,
  Settings,
  Shield,
  Users,
  X,
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { api } from '@/lib/api'

const sidebarItems = [
  { icon: LayoutDashboard, label: 'Overview', href: '/cloud' },
  { icon: Building2, label: 'Organizations', href: '/cloud/organizations' },
  { icon: Map, label: 'Sites', href: '/cloud/sites', active: true },
  { icon: HardDrive, label: 'Devices', href: '/cloud/devices' },
  { icon: BarChart3, label: 'Analytics', href: '/cloud/analytics' },
  { icon: Users, label: 'Users', href: '/cloud/users' },
  { icon: Settings, label: 'Settings', href: '/cloud/settings' },
]

interface Site {
  id: string
  name: string
  site_code: string
  country: string
  organization: string
  devices: number
}

export default function SitesPage() {
  const router = useRouter()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [sites, setSites] = useState<Site[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    if (!api.isAuthenticated()) {
      router.push('/')
      return
    }
    fetchSites()
  }, [])

  const fetchSites = async () => {
    try {
      setIsLoading(true)
      const data = await api.getSites()
      setSites(data)
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

  const getCountryName = (code: string) => {
    const countries: Record<string, string> = {
      'KE': 'Kenya',
      'TZ': 'Tanzania',
      'GH': 'Ghana',
      'NG': 'Nigeria',
      'ZA': 'South Africa',
    }
    return countries[code] || code
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
              <h2 className="text-xl font-bold text-white">Sites</h2>
              <p className="text-sm text-gray-400">Manage facility locations</p>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={fetchSites}
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

          {/* Sites Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {isLoading ? (
              [...Array(3)].map((_, i) => (
                <div key={i} className="glass-card rounded-2xl p-6 animate-pulse">
                  <div className="h-6 bg-white/10 rounded w-3/4 mb-4" />
                  <div className="h-4 bg-white/10 rounded w-1/2 mb-2" />
                  <div className="h-4 bg-white/10 rounded w-1/3" />
                </div>
              ))
            ) : sites.length === 0 ? (
              <div className="col-span-full text-center py-12 text-gray-400">
                <Map className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">No sites yet</p>
                <p className="text-sm mt-1">Sites are created when organizations are added</p>
              </div>
            ) : (
              sites.map((site, index) => (
                <motion.div
                  key={site.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="glass-card rounded-2xl p-6 hover:border-ona-primary/30 transition-all cursor-pointer"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="p-3 rounded-xl bg-blue-500/10">
                      <MapPin className="w-6 h-6 text-blue-400" />
                    </div>
                    <span className="px-2 py-1 rounded-lg text-xs font-medium bg-green-500/20 text-green-400">
                      Active
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-1">{site.name}</h3>
                  <p className="text-sm text-gray-500 mb-2">{site.site_code}</p>
                  <p className="text-sm text-gray-400 mb-4">{site.organization}</p>
                  <div className="flex items-center gap-4 text-sm">
                    <div className="flex items-center gap-1 text-gray-400">
                      <Map className="w-4 h-4" />
                      {getCountryName(site.country)}
                    </div>
                    <div className="flex items-center gap-1 text-gray-400">
                      <HardDrive className="w-4 h-4" />
                      {site.devices} devices
                    </div>
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
