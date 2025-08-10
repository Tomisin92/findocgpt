import {
    ArrowRightIcon,
    ChartBarIcon,
    DocumentTextIcon,
    SparklesIcon,
    TrendingUpIcon
} from '@heroicons/react/24/outline';
import { Link } from 'react-router-dom';

const Home = () => {
    const features = [
        {
            title: 'Document Analysis',
            description: 'Upload and analyze financial documents with AI-powered Q&A, sentiment analysis, and anomaly detection.',
            icon: DocumentTextIcon,
            href: '/documents',
            color: 'blue'
        },
        {
            title: 'Price Forecasting',
            description: 'Advanced LSTM models predict stock prices with technical indicators and confidence intervals.',
            icon: TrendingUpIcon,
            href: '/forecasting',
            color: 'green'
        },
        {
            title: 'Investment Strategy',
            description: 'Get buy/sell recommendations, portfolio optimization, and comprehensive backtesting.',
            icon: ChartBarIcon,
            href: '/strategy',
            color: 'purple'
        }
    ];

    return (
        <div className="max-w-7xl mx-auto">
            {/* Hero Section */}
            <div className="text-center mb-16">
                <div className="flex justify-center mb-6">
                    <SparklesIcon className="h-16 w-16 text-blue-600" />
                </div>

                <h1 className="text-5xl font-bold text-gray-900 mb-6">
                    Welcome to <span className="text-blue-600">FinDocGPT</span>
                </h1>

                <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
                    Transform your financial analysis with AI-powered document processing,
                    advanced forecasting models, and intelligent investment strategies.
                </p>

                <div className="flex justify-center space-x-4">
                    <Link
                        to="/documents"
                        className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium"
                    >
                        Get Started
                    </Link>
                    <Link
                        to="/dashboard"
                        className="border border-gray-300 text-gray-700 px-8 py-3 rounded-lg hover:bg-gray-50 transition-colors font-medium"
                    >
                        View Dashboard
                    </Link>
                </div>
            </div>

            {/* Features Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
                {features.map((feature, index) => {
                    const Icon = feature.icon;
                    const colorClasses = {
                        blue: 'bg-blue-50 text-blue-600 border-blue-200',
                        green: 'bg-green-50 text-green-600 border-green-200',
                        purple: 'bg-purple-50 text-purple-600 border-purple-200'
                    };

                    return (
                        <Link
                            key={index}
                            to={feature.href}
                            className="group bg-white p-8 rounded-xl shadow-md hover:shadow-lg transition-all duration-300 border border-gray-200"
                        >
                            <div className={`inline-flex p-3 rounded-lg ${colorClasses[feature.color]} mb-4`}>
                                <Icon className="h-8 w-8" />
                            </div>

                            <h3 className="text-xl font-semibold text-gray-900 mb-3 group-hover:text-blue-600 transition-colors">
                                {feature.title}
                            </h3>

                            <p className="text-gray-600 mb-4 leading-relaxed">
                                {feature.description}
                            </p>

                            <div className="flex items-center text-blue-600 font-medium group-hover:text-blue-700">
                                <span>Learn more</span>
                                <ArrowRightIcon className="h-4 w-4 ml-1 group-hover:translate-x-1 transition-transform" />
                            </div>
                        </Link>
                    );
                })}
            </div>

            {/* Stats Section */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-8 text-white">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
                    <div className="text-center">
                        <div className="text-3xl font-bold mb-2">10,231</div>
                        <div className="text-blue-100">Financial Q&A Pairs</div>
                    </div>
                    <div className="text-center">
                        <div className="text-3xl font-bold mb-2">25+</div>
                        <div className="text-blue-100">Market Indicators</div>
                    </div>
                    <div className="text-center">
                        <div className="text-3xl font-bold mb-2">3</div>
                        <div className="text-blue-100">AI-Powered Stages</div>
                    </div>
                    <div className="text-center">
                        <div className="text-3xl font-bold mb-2">Real-time</div>
                        <div className="text-blue-100">Market Data</div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Home;