/**
 */
package slco.provider;

import java.util.ArrayList;
import java.util.Collection;

import org.eclipse.emf.common.notify.Adapter;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.Notifier;

import org.eclipse.emf.edit.provider.ChangeNotifier;
import org.eclipse.emf.edit.provider.ComposeableAdapterFactory;
import org.eclipse.emf.edit.provider.ComposedAdapterFactory;
import org.eclipse.emf.edit.provider.IChangeNotifier;
import org.eclipse.emf.edit.provider.IDisposable;
import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.INotifyChangedListener;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;

import slco.util.SlcoAdapterFactory;

/**
 * This is the factory that is used to provide the interfaces needed to support Viewers.
 * The adapters generated by this factory convert EMF adapter notifications into calls to {@link #fireNotifyChanged fireNotifyChanged}.
 * The adapters also support Eclipse property sheets.
 * Note that most of the adapters are shared among multiple instances.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class SlcoItemProviderAdapterFactory extends SlcoAdapterFactory implements ComposeableAdapterFactory, IChangeNotifier, IDisposable {
	/**
	 * This keeps track of the root adapter factory that delegates to this adapter factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ComposedAdapterFactory parentAdapterFactory;

	/**
	 * This is used to implement {@link org.eclipse.emf.edit.provider.IChangeNotifier}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected IChangeNotifier changeNotifier = new ChangeNotifier();

	/**
	 * This keeps track of all the supported types checked by {@link #isFactoryForType isFactoryForType}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Collection<Object> supportedTypes = new ArrayList<Object>();

	/**
	 * This constructs an instance.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SlcoItemProviderAdapterFactory() {
		supportedTypes.add(IEditingDomainItemProvider.class);
		supportedTypes.add(IStructuredItemContentProvider.class);
		supportedTypes.add(ITreeItemContentProvider.class);
		supportedTypes.add(IItemLabelProvider.class);
		supportedTypes.add(IItemPropertySource.class);
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.ArgumentType} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ArgumentTypeItemProvider argumentTypeItemProvider;

	/**
	 * This creates an adapter for a {@link slco.ArgumentType}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createArgumentTypeAdapter() {
		if (argumentTypeItemProvider == null) {
			argumentTypeItemProvider = new ArgumentTypeItemProvider(this);
		}

		return argumentTypeItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.Assignment} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected AssignmentItemProvider assignmentItemProvider;

	/**
	 * This creates an adapter for a {@link slco.Assignment}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createAssignmentAdapter() {
		if (assignmentItemProvider == null) {
			assignmentItemProvider = new AssignmentItemProvider(this);
		}

		return assignmentItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.BidirectionalChannel} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected BidirectionalChannelItemProvider bidirectionalChannelItemProvider;

	/**
	 * This creates an adapter for a {@link slco.BidirectionalChannel}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createBidirectionalChannelAdapter() {
		if (bidirectionalChannelItemProvider == null) {
			bidirectionalChannelItemProvider = new BidirectionalChannelItemProvider(this);
		}

		return bidirectionalChannelItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.BinaryOperatorExpression} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected BinaryOperatorExpressionItemProvider binaryOperatorExpressionItemProvider;

	/**
	 * This creates an adapter for a {@link slco.BinaryOperatorExpression}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createBinaryOperatorExpressionAdapter() {
		if (binaryOperatorExpressionItemProvider == null) {
			binaryOperatorExpressionItemProvider = new BinaryOperatorExpressionItemProvider(this);
		}

		return binaryOperatorExpressionItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.BooleanConstantExpression} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected BooleanConstantExpressionItemProvider booleanConstantExpressionItemProvider;

	/**
	 * This creates an adapter for a {@link slco.BooleanConstantExpression}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createBooleanConstantExpressionAdapter() {
		if (booleanConstantExpressionItemProvider == null) {
			booleanConstantExpressionItemProvider = new BooleanConstantExpressionItemProvider(this);
		}

		return booleanConstantExpressionItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.Class} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ClassItemProvider classItemProvider;

	/**
	 * This creates an adapter for a {@link slco.Class}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createClassAdapter() {
		if (classItemProvider == null) {
			classItemProvider = new ClassItemProvider(this);
		}

		return classItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.Delay} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected DelayItemProvider delayItemProvider;

	/**
	 * This creates an adapter for a {@link slco.Delay}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createDelayAdapter() {
		if (delayItemProvider == null) {
			delayItemProvider = new DelayItemProvider(this);
		}

		return delayItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.Final} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected FinalItemProvider finalItemProvider;

	/**
	 * This creates an adapter for a {@link slco.Final}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createFinalAdapter() {
		if (finalItemProvider == null) {
			finalItemProvider = new FinalItemProvider(this);
		}

		return finalItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.Initial} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected InitialItemProvider initialItemProvider;

	/**
	 * This creates an adapter for a {@link slco.Initial}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createInitialAdapter() {
		if (initialItemProvider == null) {
			initialItemProvider = new InitialItemProvider(this);
		}

		return initialItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.IntegerConstantExpression} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected IntegerConstantExpressionItemProvider integerConstantExpressionItemProvider;

	/**
	 * This creates an adapter for a {@link slco.IntegerConstantExpression}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createIntegerConstantExpressionAdapter() {
		if (integerConstantExpressionItemProvider == null) {
			integerConstantExpressionItemProvider = new IntegerConstantExpressionItemProvider(this);
		}

		return integerConstantExpressionItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.Model} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ModelItemProvider modelItemProvider;

	/**
	 * This creates an adapter for a {@link slco.Model}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createModelAdapter() {
		if (modelItemProvider == null) {
			modelItemProvider = new ModelItemProvider(this);
		}

		return modelItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.Object} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ObjectItemProvider objectItemProvider;

	/**
	 * This creates an adapter for a {@link slco.Object}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createObjectAdapter() {
		if (objectItemProvider == null) {
			objectItemProvider = new ObjectItemProvider(this);
		}

		return objectItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.Port} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected PortItemProvider portItemProvider;

	/**
	 * This creates an adapter for a {@link slco.Port}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createPortAdapter() {
		if (portItemProvider == null) {
			portItemProvider = new PortItemProvider(this);
		}

		return portItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.SendSignal} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SendSignalItemProvider sendSignalItemProvider;

	/**
	 * This creates an adapter for a {@link slco.SendSignal}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createSendSignalAdapter() {
		if (sendSignalItemProvider == null) {
			sendSignalItemProvider = new SendSignalItemProvider(this);
		}

		return sendSignalItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.SignalArgumentExpression} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SignalArgumentExpressionItemProvider signalArgumentExpressionItemProvider;

	/**
	 * This creates an adapter for a {@link slco.SignalArgumentExpression}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createSignalArgumentExpressionAdapter() {
		if (signalArgumentExpressionItemProvider == null) {
			signalArgumentExpressionItemProvider = new SignalArgumentExpressionItemProvider(this);
		}

		return signalArgumentExpressionItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.SignalArgumentVariable} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SignalArgumentVariableItemProvider signalArgumentVariableItemProvider;

	/**
	 * This creates an adapter for a {@link slco.SignalArgumentVariable}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createSignalArgumentVariableAdapter() {
		if (signalArgumentVariableItemProvider == null) {
			signalArgumentVariableItemProvider = new SignalArgumentVariableItemProvider(this);
		}

		return signalArgumentVariableItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.SignalReception} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SignalReceptionItemProvider signalReceptionItemProvider;

	/**
	 * This creates an adapter for a {@link slco.SignalReception}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createSignalReceptionAdapter() {
		if (signalReceptionItemProvider == null) {
			signalReceptionItemProvider = new SignalReceptionItemProvider(this);
		}

		return signalReceptionItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.State} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected StateItemProvider stateItemProvider;

	/**
	 * This creates an adapter for a {@link slco.State}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createStateAdapter() {
		if (stateItemProvider == null) {
			stateItemProvider = new StateItemProvider(this);
		}

		return stateItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.StateMachine} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected StateMachineItemProvider stateMachineItemProvider;

	/**
	 * This creates an adapter for a {@link slco.StateMachine}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createStateMachineAdapter() {
		if (stateMachineItemProvider == null) {
			stateMachineItemProvider = new StateMachineItemProvider(this);
		}

		return stateMachineItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.StringConstantExpression} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected StringConstantExpressionItemProvider stringConstantExpressionItemProvider;

	/**
	 * This creates an adapter for a {@link slco.StringConstantExpression}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createStringConstantExpressionAdapter() {
		if (stringConstantExpressionItemProvider == null) {
			stringConstantExpressionItemProvider = new StringConstantExpressionItemProvider(this);
		}

		return stringConstantExpressionItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.TextualStatement} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected TextualStatementItemProvider textualStatementItemProvider;

	/**
	 * This creates an adapter for a {@link slco.TextualStatement}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createTextualStatementAdapter() {
		if (textualStatementItemProvider == null) {
			textualStatementItemProvider = new TextualStatementItemProvider(this);
		}

		return textualStatementItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.Transition} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected TransitionItemProvider transitionItemProvider;

	/**
	 * This creates an adapter for a {@link slco.Transition}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createTransitionAdapter() {
		if (transitionItemProvider == null) {
			transitionItemProvider = new TransitionItemProvider(this);
		}

		return transitionItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.UnidirectionalChannel} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected UnidirectionalChannelItemProvider unidirectionalChannelItemProvider;

	/**
	 * This creates an adapter for a {@link slco.UnidirectionalChannel}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createUnidirectionalChannelAdapter() {
		if (unidirectionalChannelItemProvider == null) {
			unidirectionalChannelItemProvider = new UnidirectionalChannelItemProvider(this);
		}

		return unidirectionalChannelItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.Variable} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected VariableItemProvider variableItemProvider;

	/**
	 * This creates an adapter for a {@link slco.Variable}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createVariableAdapter() {
		if (variableItemProvider == null) {
			variableItemProvider = new VariableItemProvider(this);
		}

		return variableItemProvider;
	}

	/**
	 * This keeps track of the one adapter used for all {@link slco.VariableExpression} instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected VariableExpressionItemProvider variableExpressionItemProvider;

	/**
	 * This creates an adapter for a {@link slco.VariableExpression}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter createVariableExpressionAdapter() {
		if (variableExpressionItemProvider == null) {
			variableExpressionItemProvider = new VariableExpressionItemProvider(this);
		}

		return variableExpressionItemProvider;
	}

	/**
	 * This returns the root adapter factory that contains this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComposeableAdapterFactory getRootAdapterFactory() {
		return parentAdapterFactory == null ? this : parentAdapterFactory.getRootAdapterFactory();
	}

	/**
	 * This sets the composed adapter factory that contains this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setParentAdapterFactory(ComposedAdapterFactory parentAdapterFactory) {
		this.parentAdapterFactory = parentAdapterFactory;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean isFactoryForType(Object type) {
		return supportedTypes.contains(type) || super.isFactoryForType(type);
	}

	/**
	 * This implementation substitutes the factory itself as the key for the adapter.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Adapter adapt(Notifier notifier, Object type) {
		return super.adapt(notifier, this);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object adapt(Object object, Object type) {
		if (isFactoryForType(type)) {
			Object adapter = super.adapt(object, type);
			if (!(type instanceof Class<?>) || (((Class<?>)type).isInstance(adapter))) {
				return adapter;
			}
		}

		return null;
	}

	/**
	 * This adds a listener.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void addListener(INotifyChangedListener notifyChangedListener) {
		changeNotifier.addListener(notifyChangedListener);
	}

	/**
	 * This removes a listener.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void removeListener(INotifyChangedListener notifyChangedListener) {
		changeNotifier.removeListener(notifyChangedListener);
	}

	/**
	 * This delegates to {@link #changeNotifier} and to {@link #parentAdapterFactory}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void fireNotifyChanged(Notification notification) {
		changeNotifier.fireNotifyChanged(notification);

		if (parentAdapterFactory != null) {
			parentAdapterFactory.fireNotifyChanged(notification);
		}
	}

	/**
	 * This disposes all of the item providers created by this factory. 
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void dispose() {
		if (argumentTypeItemProvider != null) argumentTypeItemProvider.dispose();
		if (assignmentItemProvider != null) assignmentItemProvider.dispose();
		if (bidirectionalChannelItemProvider != null) bidirectionalChannelItemProvider.dispose();
		if (binaryOperatorExpressionItemProvider != null) binaryOperatorExpressionItemProvider.dispose();
		if (booleanConstantExpressionItemProvider != null) booleanConstantExpressionItemProvider.dispose();
		if (classItemProvider != null) classItemProvider.dispose();
		if (delayItemProvider != null) delayItemProvider.dispose();
		if (finalItemProvider != null) finalItemProvider.dispose();
		if (initialItemProvider != null) initialItemProvider.dispose();
		if (integerConstantExpressionItemProvider != null) integerConstantExpressionItemProvider.dispose();
		if (modelItemProvider != null) modelItemProvider.dispose();
		if (objectItemProvider != null) objectItemProvider.dispose();
		if (portItemProvider != null) portItemProvider.dispose();
		if (sendSignalItemProvider != null) sendSignalItemProvider.dispose();
		if (signalArgumentExpressionItemProvider != null) signalArgumentExpressionItemProvider.dispose();
		if (signalArgumentVariableItemProvider != null) signalArgumentVariableItemProvider.dispose();
		if (signalReceptionItemProvider != null) signalReceptionItemProvider.dispose();
		if (stateItemProvider != null) stateItemProvider.dispose();
		if (stateMachineItemProvider != null) stateMachineItemProvider.dispose();
		if (stringConstantExpressionItemProvider != null) stringConstantExpressionItemProvider.dispose();
		if (textualStatementItemProvider != null) textualStatementItemProvider.dispose();
		if (transitionItemProvider != null) transitionItemProvider.dispose();
		if (unidirectionalChannelItemProvider != null) unidirectionalChannelItemProvider.dispose();
		if (variableItemProvider != null) variableItemProvider.dispose();
		if (variableExpressionItemProvider != null) variableExpressionItemProvider.dispose();
	}

}
