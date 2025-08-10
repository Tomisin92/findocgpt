
import { BriefcaseIcon, ChartBarIcon, DocumentTextIcon, TrendingUpIcon } from '@heroicons/react/24/outline';
import { Link, useLocation } from 'react-router-dom';

const Header = () => {
    const location = useLocation();

    const navigation = [
        { name: 'Dashboard', href: '/dashboard', icon: ChartBarIcon },
        { name: 'Documents', href: '/documents', icon: DocumentTextIcon },
        { name: 'Forecasting', href: '/forecasting', icon: TrendingUpIcon },
        { name: 'Strategy', href: '/strategy', icon: BriefcaseIcon },
    ];

    return (
        <header className="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg">
            <div className="container mx-auto px-4 py-4">
                <div className="flex justify-between items-center">
                    <Link to="/" className="text-2xl font-bold">
                        FinDocGPT
                    </Link>

                    <nav className="flex space-x-6">
                        {navigation.map((item) => {
                            const Icon = item.icon;
                            const isActive = location.pathname === item.href;

                            return (
                                <Link
                                    key={item.name}
                                    to={item.href}
                                    className={`flex items-center space-x-2 px-3 py-2 rounded-md transition-colors ${isActive
                                            ? 'bg-white/20 text-white'
                                            : 'hover:bg-white/10 text-blue-100'
                                        }`}
                                >
                                    <Icon className="h-5 w-5" />
                                    <span>{item.name}</span>
                                </Link>
                            );
                        })}
                    </nav>
                </div>
            </div>
        </header>
    );
};

export default Header;