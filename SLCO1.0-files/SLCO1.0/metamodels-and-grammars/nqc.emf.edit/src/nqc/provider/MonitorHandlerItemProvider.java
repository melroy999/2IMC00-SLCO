/**
 */
package nqc.provider;


import java.util.Collection;
import java.util.List;

import nqc.MonitorHandler;
import nqc.NqcFactory;
import nqc.NqcPackage;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.common.util.ResourceLocator;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ItemProviderAdapter;
import org.eclipse.emf.edit.provider.ViewerNotification;

/**
 * This is the item provider adapter for a {@link nqc.MonitorHandler} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class MonitorHandlerItemProvider
	extends ItemProviderAdapter
	implements
		IEditingDomainItemProvider,
		IStructuredItemContentProvider,
		ITreeItemContentProvider,
		IItemLabelProvider,
		IItemPropertySource {
	/**
	 * This constructs an instance from a factory and a notifier.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MonitorHandlerItemProvider(AdapterFactory adapterFactory) {
		super(adapterFactory);
	}

	/**
	 * This returns the property descriptors for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public List<IItemPropertyDescriptor> getPropertyDescriptors(Object object) {
		if (itemPropertyDescriptors == null) {
			super.getPropertyDescriptors(object);

		}
		return itemPropertyDescriptors;
	}

	/**
	 * This specifies how to implement {@link #getChildren} and is used to deduce an appropriate feature for an
	 * {@link org.eclipse.emf.edit.command.AddCommand}, {@link org.eclipse.emf.edit.command.RemoveCommand} or
	 * {@link org.eclipse.emf.edit.command.MoveCommand} in {@link #createCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Collection<? extends EStructuralFeature> getChildrenFeatures(Object object) {
		if (childrenFeatures == null) {
			super.getChildrenFeatures(object);
			childrenFeatures.add(NqcPackage.eINSTANCE.getMonitorHandler_CatchEvent());
			childrenFeatures.add(NqcPackage.eINSTANCE.getMonitorHandler_Handler());
		}
		return childrenFeatures;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EStructuralFeature getChildFeature(Object object, Object child) {
		// Check the type of the specified child object and return the proper feature to use for
		// adding (see {@link AddCommand}) it as a child.

		return super.getChildFeature(object, child);
	}

	/**
	 * This returns MonitorHandler.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/MonitorHandler"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		return getString("_UI_MonitorHandler_type");
	}

	/**
	 * This handles model notifications by calling {@link #updateChildren} to update any cached
	 * children and by creating a viewer notification, which it passes to {@link #fireNotifyChanged}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void notifyChanged(Notification notification) {
		updateChildren(notification);

		switch (notification.getFeatureID(MonitorHandler.class)) {
			case NqcPackage.MONITOR_HANDLER__CATCH_EVENT:
			case NqcPackage.MONITOR_HANDLER__HANDLER:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), true, false));
				return;
		}
		super.notifyChanged(notification);
	}

	/**
	 * This adds {@link org.eclipse.emf.edit.command.CommandParameter}s describing the children
	 * that can be created under this object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected void collectNewChildDescriptors(Collection<Object> newChildDescriptors, Object object) {
		super.collectNewChildDescriptors(newChildDescriptors, object);

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_CatchEvent(),
				 NqcFactory.eINSTANCE.createEventTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createAcquireConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createAcquireStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createVariableExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createArrayExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createAssignmentStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createBinaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createBinaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createBlockStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createBooleanConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createBreakStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createContinueStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createDirectionConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createDisplayModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createDoWhileStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createEmptyStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createEventTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createForStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createGoToStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createIfStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createIntegerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createMonitorStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createNullaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createOutputModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createOutputPortNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createQuaternaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createRepeatStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createReturnStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSenaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSensorConfigConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSensorModeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSensorNameConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSensorTypeConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSerialBaudConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSerialBiphaseConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSerialChecksumConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSerialChannelConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSerialCommConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSerialPacketConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSoundConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createStartStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createStopStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSubroutineCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createSwitchStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createTernaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createTernaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createTxPowerConstant()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createUnaryBuiltInValueFunctionCall()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createUnaryExpression()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createUntilStatement()));

		newChildDescriptors.add
			(createChildParameter
				(NqcPackage.eINSTANCE.getMonitorHandler_Handler(),
				 NqcFactory.eINSTANCE.createWhileStatement()));
	}

	/**
	 * This returns the label text for {@link org.eclipse.emf.edit.command.CreateChildCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getCreateChildText(Object owner, Object feature, Object child, Collection<?> selection) {
		Object childFeature = feature;
		Object childObject = child;

		boolean qualify =
			childFeature == NqcPackage.eINSTANCE.getMonitorHandler_CatchEvent() ||
			childFeature == NqcPackage.eINSTANCE.getMonitorHandler_Handler();

		if (qualify) {
			return getString
				("_UI_CreateChild_text2",
				 new Object[] { getTypeText(childObject), getFeatureText(childFeature), getTypeText(owner) });
		}
		return super.getCreateChildText(owner, feature, child, selection);
	}

	/**
	 * Return the resource locator for this item provider's resources.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public ResourceLocator getResourceLocator() {
		return NqcEditPlugin.INSTANCE;
	}

}
