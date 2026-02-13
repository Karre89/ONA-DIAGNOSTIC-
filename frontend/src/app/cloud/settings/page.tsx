'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  AlertTriangle,
  BarChart3,
  Building2,
  Check,
  HardDrive,
  Key,
  LayoutDashboard,
  LogOut,
  Map,
  RefreshCw,
  Settings,
  Shield,
  User,
  Users,
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { api } from '@/lib/api'

const sidebarItems = [
  { icon: LayoutDashboard, label: 'Overview', href: '/cloud' },
  { icon: Building2, label: 'Organizations', href: '/cloud/organizations' },
  { icon: Map, label: 'Sites', href: '/cloud/sites' },
  { icon: HardDrive, label: 'Devices', href: '/cloud/devices' },
  { icon: BarChart3, label: 'Analytics', href: '/cloud/analytics' },
  { icon: Users, label: 'Users', href: '/cloud/users' },
  { icon: Settings, label: 'Settings', href: '/cloud/settings', active: true },
]

export default function SettingsPage() {
  const router = useRouter()
  const [sidebarOpen] = useState(true)
  const [user, setUser] = useState<any>(null)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  // Profile form
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [profileSaving, setProfileSaving] = useState(false)

  // Password form
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [passwordSaving, setPasswordSaving] = useState(false)
  const [passwordError, setPasswordError] = useState('')
  const [passwordSuccess, setPasswordSuccess] = useState('')

  useEffect(() => {
    if (!api.isAuthenticated()) {
      router.push('/')
      return
    }

    const userStr = localStorage.getItem('user')
    if (userStr) {
      const u = JSON.parse(userStr)
      setUser(u)
      setFullName(u.full_name || '')
      setEmail(u.email || '')
    }
  }, [])

  const handleLogout = () => {
    api.logout()
    router.push('/')
  }

  const handleProfileSave = async () => {
    if (!user) return
    try {
      setProfileSaving(true)
      setError('')
      setSuccess('')
      await api.updateUser(user.id, { full_name: fullName })
      // Update localStorage
      const updatedUser = { ...user, full_name: fullName }
      localStorage.setItem('user', JSON.stringify(updatedUser))
      setUser(updatedUser)
      setSuccess('Profile updated successfully')
      setTimeout(() => setSuccess(''), 3000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update profile')
    } finally {
      setProfileSaving(false)
    }
  }

  const handlePasswordChange = async () => {
    setPasswordError('')
    setPasswordSuccess('')

    if (!newPassword || !currentPassword) {
      setPasswordError('All fields are required')
      return
    }
    if (newPassword.length < 6) {
      setPasswordError('New password must be at least 6 characters')
      return
    }
    if (newPassword !== confirmPassword) {
      setPasswordError('New passwords do not match')
      return
    }

    try {
      setPasswordSaving(true)
      await api.changePassword(currentPassword, newPassword)
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
      setPasswordSuccess('Password changed successfully')
      setTimeout(() => setPasswordSuccess(''), 3000)
    } catch (err) {
      setPasswordError(err instanceof Error ? err.message : 'Failed to change password')
    } finally {
      setPasswordSaving(false)
    }
  }

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className={`glass-strong border-r border-white/5 transition-all duration-300 ${sidebarOpen ? 'w-64' : 'w-20'}`}>
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
            <Link key={item.label} href={item.href}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all ${
                item.active ? 'bg-ona-primary/20 text-ona-primary' : 'text-gray-400 hover:bg-white/5 hover:text-white'
              }`}>
              <item.icon className="w-5 h-5 flex-shrink-0" />
              {sidebarOpen && <span>{item.label}</span>}
            </Link>
          ))}
        </nav>
        <div className="absolute bottom-4 left-4 right-4">
          <button onClick={handleLogout} className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-gray-400 hover:bg-white/5 hover:text-white transition-all">
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
              <h2 className="text-xl font-bold text-white">Settings</h2>
              <p className="text-sm text-gray-400">Manage your account and preferences</p>
            </div>
          </div>
        </header>

        <div className="p-8 max-w-2xl space-y-6">
          {error && (
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
              className="p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 flex items-center gap-3">
              <AlertTriangle className="w-5 h-5" />
              <span>{error}</span>
            </motion.div>
          )}

          {success && (
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
              className="p-4 rounded-xl bg-green-500/10 border border-green-500/30 text-green-400 flex items-center gap-3">
              <Check className="w-5 h-5" />
              <span>{success}</span>
            </motion.div>
          )}

          {/* Profile Section */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="glass-card rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-6">
              <User className="w-5 h-5 text-ona-primary" />
              <h3 className="text-lg font-semibold text-white">Profile</h3>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Full Name</label>
                <input type="text" value={fullName} onChange={e => setFullName(e.target.value)}
                  className="w-full glass rounded-xl px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-ona-primary" />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Email</label>
                <input type="email" value={email} disabled
                  className="w-full glass rounded-xl px-4 py-2.5 text-gray-500 cursor-not-allowed" />
                <p className="text-xs text-gray-600 mt-1">Email cannot be changed</p>
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Role</label>
                <div className="glass rounded-xl px-4 py-2.5 text-gray-400 capitalize">{user?.role || '—'}</div>
              </div>

              <button onClick={handleProfileSave} disabled={profileSaving}
                className="btn-primary py-2.5 px-6 rounded-xl flex items-center gap-2">
                {profileSaving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                Save Changes
              </button>
            </div>
          </motion.div>

          {/* Password Section */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
            className="glass-card rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-6">
              <Key className="w-5 h-5 text-ona-primary" />
              <h3 className="text-lg font-semibold text-white">Change Password</h3>
            </div>

            {passwordError && (
              <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">{passwordError}</div>
            )}
            {passwordSuccess && (
              <div className="mb-4 p-3 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400 text-sm">{passwordSuccess}</div>
            )}

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Current Password</label>
                <input type="password" value={currentPassword} onChange={e => setCurrentPassword(e.target.value)}
                  className="w-full glass rounded-xl px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-ona-primary"
                  placeholder="Enter current password" />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">New Password</label>
                <input type="password" value={newPassword} onChange={e => setNewPassword(e.target.value)}
                  className="w-full glass rounded-xl px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-ona-primary"
                  placeholder="Minimum 6 characters" />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Confirm New Password</label>
                <input type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)}
                  className="w-full glass rounded-xl px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-ona-primary"
                  placeholder="Re-enter new password" />
              </div>

              <button onClick={handlePasswordChange} disabled={passwordSaving}
                className="btn-primary py-2.5 px-6 rounded-xl flex items-center gap-2">
                {passwordSaving ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Key className="w-4 h-4" />}
                Change Password
              </button>
            </div>
          </motion.div>

          {/* Platform Info */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            className="glass-card rounded-2xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <Shield className="w-5 h-5 text-ona-primary" />
              <h3 className="text-lg font-semibold text-white">Platform</h3>
            </div>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Version</span>
                <span className="text-white">0.1.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">API</span>
                <span className="text-white font-mono text-xs">ona-diagnostic-production.up.railway.app</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Auth</span>
                <span className="text-white">JWT + pbkdf2_sha256</span>
              </div>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  )
}
