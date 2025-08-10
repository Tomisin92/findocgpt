import {
    BriefcaseIcon,
    ChartBarIcon,
    DocumentTextIcon,
    HomeIcon,
    TrendingUpIcon
} from '@heroicons/react/24/outline';
import { Link, useLocation } from 'react-router-dom';

const Sidebar = () => {
    const location = useLocation();

    const navigation = [
        { name: 'Home', href: '/', icon: HomeIcon },
        { name: 'Document Analysis', href: '/documents', icon: DocumentTextIcon },
        { name: 'Price Forecasting', href: '/forecasting', icon: TrendingUpIcon },
        { name: 'Investment Strategy', href: '/strategy', icon: BriefcaseIcon },
        { name: 'Dashboard', href: '/dashboard', icon: ChartBarIcon },
    ];

    return (
        <div className="w-64 bg-white shadow-lg h-screen">
            <div className="p-4">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Navigation</h2>

                <nav className="space-y-2">
                    {navigation.map((item) => {
                        const Icon = item.icon;
                        const isActive = location.pathname === item.href;

                        return (
                            <Link
                                key={item.name}
                                to={item.href}
                                className={`flex items-center space-x-3 px-3 py-2 rounded-md transition-colors ${isActive
                                        ? 'bg-blue-50 text-blue-600 border-r-2 border-blue-600'
                                        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                                    }`}
                            >
                                <Icon className="h-5 w-5" />
                                <span className="text-sm font-medium">{item.name}</span>
                            </Link>
                        );
                    })}
                </nav>
            </div>

            <div className="absolute bottom-4 left-4 right-4">
                <div className="bg-blue-50 p-3 rounded-lg">
                    <p className="text-xs text-blue-600 font-medium">FinDocGPT v1.0</p>
                    <p className="text-xs text-blue-500">AI-Powered Financial Analysis</p>
                </div>
            </div>
        </div>
    );
};

export default Sidebar;
