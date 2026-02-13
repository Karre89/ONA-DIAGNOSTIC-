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
  Plus,
  RefreshCw,
  Settings,
  Shield,
  Users,
  UserPlus,
  X,
  Check,
  Ban,
  Trash2,
} from 'lucide-react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { api, UserInfo } from '@/lib/api'

const sidebarItems = [
  { icon: LayoutDashboard, label: 'Overview', href: '/cloud' },
  { icon: Building2, label: 'Organizations', href: '/cloud/organizations' },
  { icon: Map, label: 'Sites', href: '/cloud/sites' },
  { icon: HardDrive, label: 'Devices', href: '/cloud/devices' },
  { icon: BarChart3, label: 'Analytics', href: '/cloud/analytics' },
  { icon: Users, label: 'Users', href: '/cloud/users', active: true },
  { icon: Settings, label: 'Settings', href: '/cloud/settings' },
]

const roleColors: Record<string, string> = {
  admin: 'bg-purple-500/20 text-purple-400',
  clinic_user: 'bg-blue-500/20 text-blue-400',
  viewer: 'bg-gray-500/20 text-gray-400',
}

export default function UsersPage() {
  const router = useRouter()
  const [sidebarOpen] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [users, setUsers] = useState<UserInfo[]>([])
  const [error, setError] = useState('')
  const [showAddModal, setShowAddModal] = useState(false)
  const [newUser, setNewUser] = useState({ email: '', password: '', full_name: '', role: 'clinic_user' })
  const [addError, setAddError] = useState('')

  useEffect(() => {
    if (!api.isAuthenticated()) {
      router.push('/')
      return
    }
    fetchUsers()
  }, [])

  const fetchUsers = async () => {
    try {
      setIsLoading(true)
      const data = await api.getUsers()
      setUsers(data)
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load users')
    } finally {
      setIsLoading(false)
    }
  }

  const handleLogout = () => {
    api.logout()
    router.push('/')
  }

  const handleAddUser = async () => {
    if (!newUser.email || !newUser.password || !newUser.full_name) {
      setAddError('All fields are required')
      return
    }
    try {
      setAddError('')
      await api.createUser(newUser)
      setShowAddModal(false)
      setNewUser({ email: '', password: '', full_name: '', role: 'clinic_user' })
      fetchUsers()
    } catch (err) {
      setAddError(err instanceof Error ? err.message : 'Failed to create user')
    }
  }

  const handleToggleActive = async (user: UserInfo) => {
    try {
      await api.updateUser(user.id, { is_active: !user.is_active })
      fetchUsers()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update user')
    }
  }

  const handleDeleteUser = async (user: UserInfo) => {
    if (!confirm(`Delete ${user.full_name} (${user.email})?`)) return
    try {
      await api.deleteUser(user.id)
      fetchUsers()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete user')
    }
  }

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return 'Never'
    const d = new Date(dateStr)
    const now = new Date()
    const diff = now.getTime() - d.getTime()
    const days = Math.floor(diff / 86400000)
    if (days === 0) return 'Today'
    if (days === 1) return 'Yesterday'
    if (days < 30) return `${days}d ago`
    return d.toLocaleDateString()
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
              <h2 className="text-xl font-bold text-white">Users</h2>
              <p className="text-sm text-gray-400">Manage platform users and roles</p>
            </div>
            <div className="flex items-center gap-3">
              <button onClick={fetchUsers} disabled={isLoading} className="p-2 rounded-lg hover:bg-white/5 transition-colors">
                <RefreshCw className={`w-5 h-5 text-gray-400 ${isLoading ? 'animate-spin' : ''}`} />
              </button>
              <button onClick={() => setShowAddModal(true)} className="btn-primary text-sm py-2 px-4 flex items-center gap-2">
                <UserPlus className="w-4 h-4" /> Add User
              </button>
            </div>
          </div>
        </header>

        <div className="p-8">
          {error && (
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
              className="mb-6 p-4 rounded-xl bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 flex items-center gap-3">
              <AlertTriangle className="w-5 h-5" />
              <span>{error}</span>
            </motion.div>
          )}

          {/* Stats row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            {[
              { label: 'Total Users', value: users.length },
              { label: 'Active', value: users.filter(u => u.is_active).length },
              { label: 'Admins', value: users.filter(u => u.role === 'admin').length },
            ].map((stat, i) => (
              <motion.div key={stat.label} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}
                className="glass-card rounded-2xl p-5">
                <p className="text-3xl font-bold text-white mb-1">{isLoading ? '-' : stat.value}</p>
                <p className="text-sm text-gray-400">{stat.label}</p>
              </motion.div>
            ))}
          </div>

          {/* Users Table */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
            className="glass-card rounded-2xl p-6">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">User</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Role</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Organization</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Status</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Last Login</th>
                    <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {isLoading ? (
                    [...Array(3)].map((_, i) => (
                      <tr key={i} className="border-b border-white/5">
                        <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-40 animate-pulse" /></td>
                        <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-20 animate-pulse" /></td>
                        <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-24 animate-pulse" /></td>
                        <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-16 animate-pulse" /></td>
                        <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-16 animate-pulse" /></td>
                        <td className="py-4 px-4"><div className="h-4 bg-white/10 rounded w-20 animate-pulse" /></td>
                      </tr>
                    ))
                  ) : users.length === 0 ? (
                    <tr><td colSpan={6} className="py-8 text-center text-gray-400">No users yet.</td></tr>
                  ) : (
                    users.map((user, index) => (
                      <motion.tr key={user.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + index * 0.05 }}
                        className="border-b border-white/5 hover:bg-white/5">
                        <td className="py-4 px-4">
                          <p className="text-white font-medium">{user.full_name}</p>
                          <p className="text-xs text-gray-500">{user.email}</p>
                        </td>
                        <td className="py-4 px-4">
                          <span className={`px-2 py-1 rounded-lg text-xs font-medium ${roleColors[user.role] || roleColors.viewer}`}>
                            {user.role}
                          </span>
                        </td>
                        <td className="py-4 px-4 text-gray-400 text-sm">
                          {user.tenant_name || '—'}
                          {user.site_name && <span className="text-gray-600 ml-1">/ {user.site_name}</span>}
                        </td>
                        <td className="py-4 px-4">
                          <span className={`px-2 py-1 rounded-lg text-xs font-medium ${
                            user.is_active ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                          }`}>
                            {user.is_active ? 'Active' : 'Disabled'}
                          </span>
                        </td>
                        <td className="py-4 px-4 text-gray-400 text-sm">{formatDate(user.last_login)}</td>
                        <td className="py-4 px-4">
                          <div className="flex items-center gap-2">
                            <button onClick={() => handleToggleActive(user)}
                              title={user.is_active ? 'Disable' : 'Enable'}
                              className="p-1.5 rounded-lg hover:bg-white/10 transition-colors">
                              {user.is_active ?
                                <Ban className="w-4 h-4 text-yellow-400" /> :
                                <Check className="w-4 h-4 text-green-400" />
                              }
                            </button>
                            <button onClick={() => handleDeleteUser(user)}
                              title="Delete"
                              className="p-1.5 rounded-lg hover:bg-white/10 transition-colors">
                              <Trash2 className="w-4 h-4 text-red-400" />
                            </button>
                          </div>
                        </td>
                      </motion.tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>
      </main>

      {/* Add User Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
            className="glass-card rounded-2xl p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-white">Add New User</h3>
              <button onClick={() => { setShowAddModal(false); setAddError('') }} className="p-1 rounded-lg hover:bg-white/10">
                <X className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            {addError && (
              <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">{addError}</div>
            )}

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Full Name</label>
                <input type="text" value={newUser.full_name} onChange={e => setNewUser({...newUser, full_name: e.target.value})}
                  className="w-full glass rounded-xl px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-ona-primary"
                  placeholder="John Doe" />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Email</label>
                <input type="email" value={newUser.email} onChange={e => setNewUser({...newUser, email: e.target.value})}
                  className="w-full glass rounded-xl px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-ona-primary"
                  placeholder="user@ona.health" />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Password</label>
                <input type="password" value={newUser.password} onChange={e => setNewUser({...newUser, password: e.target.value})}
                  className="w-full glass rounded-xl px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-ona-primary"
                  placeholder="Minimum 6 characters" />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Role</label>
                <select value={newUser.role} onChange={e => setNewUser({...newUser, role: e.target.value})}
                  className="w-full glass rounded-xl px-4 py-2.5 text-white bg-transparent focus:outline-none focus:ring-1 focus:ring-ona-primary">
                  <option value="clinic_user" className="bg-gray-900">Clinic User</option>
                  <option value="admin" className="bg-gray-900">Admin</option>
                  <option value="viewer" className="bg-gray-900">Viewer</option>
                </select>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button onClick={() => { setShowAddModal(false); setAddError('') }}
                className="flex-1 glass py-2.5 rounded-xl text-gray-300 hover:text-white transition-colors">Cancel</button>
              <button onClick={handleAddUser}
                className="flex-1 btn-primary py-2.5 rounded-xl">Create User</button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}
